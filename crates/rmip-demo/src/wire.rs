use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{Coord, Symbol};
use rmip_iop::driver::{Execution, Occurrence, DUMMY_ORACLE};
use rmip_iop::transcript::{Entry, OracleId, Transcript, TranscriptPrefix};
use serde_json::{json, Value};
use std::io::{self, Write};

pub const VERSION: u8 = 1;
pub const VERSION_WIDE: u8 = 2;
pub const VERSION_WIDE256: u8 = 3;
pub const VERSION_WIDE1024: u8 = 4;
pub const ELEMENT_BYTES: usize = 16;
const MAGIC: &[u8; 4] = b"RMLW";

fn coord_width(domain: Option<Coord>) -> u8 {
    match domain {
        None => 8,
        Some(d) if d <= Coord::pow2(64) => 8,
        Some(d) if d <= Coord::pow2(128) => 16,
        Some(d) if d <= Coord::pow2(256) => 32,
        Some(_) => 128,
    }
}

fn coord_bytes(version: u8) -> u8 {
    match version {
        VERSION_WIDE1024 => 128,
        VERSION_WIDE256 => 32,
        VERSION_WIDE => 16,
        _ => 8,
    }
}

fn version_for(width: u8) -> u8 {
    match width {
        8 => VERSION,
        16 => VERSION_WIDE,
        32 => VERSION_WIDE256,
        _ => VERSION_WIDE1024,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    MainOrdinary = 1,
    PublicCoin = 2,
    MainOraclePost = 3,
    PublicOraclePost = 4,
    MainQuery = 5,
    MainAnswer = 6,
    CheckerQuery = 7,
    CheckerAnswer = 8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommunicationBytes {
    pub main_to_verifier_bytes: u64,
    pub verifier_to_main_bytes: u64,
    pub verifier_to_checker_bytes: u64,
    pub checker_to_verifier_bytes: u64,
    pub payload_bytes: u64,
    pub framing_bytes: u64,
    pub total_bytes: u64,
    pub frames: u64,
    pub wide_frames: u64,
    pub wide256_frames: u64,
    pub wide1024_frames: u64,
}

impl CommunicationBytes {
    pub fn widest_version(self) -> u8 {
        if self.wide1024_frames > 0 {
            VERSION_WIDE1024
        } else if self.wide256_frames > 0 {
            VERSION_WIDE256
        } else if self.wide_frames > 0 {
            VERSION_WIDE
        } else {
            VERSION
        }
    }

    pub fn widest_coordinate_bytes(self) -> u8 {
        coord_bytes(self.widest_version())
    }

    pub fn total_bytes(self) -> u64 {
        self.total_bytes
    }

    pub fn checked_add(self, other: Self) -> io::Result<Self> {
        Ok(Self {
            main_to_verifier_bytes: add(self.main_to_verifier_bytes, other.main_to_verifier_bytes)?,
            verifier_to_main_bytes: add(self.verifier_to_main_bytes, other.verifier_to_main_bytes)?,
            verifier_to_checker_bytes: add(
                self.verifier_to_checker_bytes,
                other.verifier_to_checker_bytes,
            )?,
            checker_to_verifier_bytes: add(
                self.checker_to_verifier_bytes,
                other.checker_to_verifier_bytes,
            )?,
            payload_bytes: add(self.payload_bytes, other.payload_bytes)?,
            framing_bytes: add(self.framing_bytes, other.framing_bytes)?,
            total_bytes: add(self.total_bytes, other.total_bytes)?,
            frames: add(self.frames, other.frames)?,
            wide_frames: add(self.wide_frames, other.wide_frames)?,
            wide256_frames: add(self.wide256_frames, other.wide256_frames)?,
            wide1024_frames: add(self.wide1024_frames, other.wide1024_frames)?,
        })
    }

    pub fn to_json(self) -> Value {
        json!({
            "serialization": format!("rmip-wire-v{}", self.widest_version()),
            "version": self.widest_version(),
            "measurement": "bytes successfully written by the canonical serializer",
            "main_to_verifier_bytes": self.main_to_verifier_bytes,
            "verifier_to_main_bytes": self.verifier_to_main_bytes,
            "verifier_to_checker_bytes": self.verifier_to_checker_bytes,
            "checker_to_verifier_bytes": self.checker_to_verifier_bytes,
            "payload_bytes": self.payload_bytes, "framing_bytes": self.framing_bytes,
            "total_bytes": self.total_bytes, "frames": self.frames,
            "transport": "in_process_serialization; TCP/TLS and OS headers excluded",
            "public_inputs": "fixed workload and retained preprocessing shared before online execution",
            "audit_index_transmitted": false,
        })
    }
}

impl std::ops::AddAssign for CommunicationBytes {
    fn add_assign(&mut self, other: Self) {
        *self = self.checked_add(other).expect("wire byte count overflow");
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn add(a: u64, b: u64) -> io::Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("wire byte count overflow"))
}

fn len(n: usize) -> io::Result<u64> {
    u64::try_from(n).map_err(|_| invalid("wire length exceeds u64"))
}

struct Encoder<W> {
    sink: W,
    written: u64,
    counts: CommunicationBytes,
}

impl<W: Write> Write for Encoder<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = self.sink.write(bytes)?;
        if n > bytes.len() {
            return Err(invalid("wire sink returned an invalid write length"));
        }
        self.written = add(self.written, len(n)?)?;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sink.flush()
    }
}

impl<W: Write> Encoder<W> {
    fn u64(&mut self, value: u64) -> io::Result<()> {
        self.write_all(&value.to_be_bytes())
    }

    fn coord(&mut self, width: u8, coord: Coord) -> io::Result<()> {
        match width {
            128 => self.write_all(&coord.to_be_bytes()),
            32 => {
                if !coord.fits_u256() {
                    return Err(invalid(
                        "wire coordinate exceeds 256 bits in a thirty-two-byte frame",
                    ));
                }
                self.write_all(&coord.to_be_bytes()[96..])
            }
            16 => {
                let wide = u128::try_from(coord)
                    .map_err(|_| invalid("wire coordinate exceeds u128 in a sixteen-byte frame"))?;
                self.write_all(&wide.to_be_bytes())
            }
            _ => {
                let narrow = u64::try_from(coord)
                    .map_err(|_| invalid("wire coordinate exceeds u64 in a narrow domain"))?;
                self.u64(narrow)
            }
        }
    }

    fn frame(
        &mut self,
        kind: Kind,
        payload_len: u64,
        body: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        self.frame_at(VERSION, kind, payload_len, body)
    }

    fn frame_at(
        &mut self,
        version: u8,
        kind: Kind,
        payload_len: u64,
        body: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        let before = self.written;
        self.write_all(MAGIC)?;
        self.write_all(&[version, kind as u8])?;
        self.u64(payload_len)?;
        let payload_start = self.written;
        body(self)?;
        let payload = self.written - payload_start;
        if payload != payload_len {
            return Err(invalid("wire payload does not match its declared length"));
        }
        let count = self.written - before;
        let direction = match kind {
            Kind::MainOrdinary | Kind::MainOraclePost | Kind::MainAnswer => {
                &mut self.counts.main_to_verifier_bytes
            }
            Kind::PublicCoin | Kind::PublicOraclePost | Kind::MainQuery => {
                &mut self.counts.verifier_to_main_bytes
            }
            Kind::CheckerQuery => &mut self.counts.verifier_to_checker_bytes,
            Kind::CheckerAnswer => &mut self.counts.checker_to_verifier_bytes,
        };
        *direction = add(*direction, count)?;
        self.counts.framing_bytes = add(self.counts.framing_bytes, payload_start - before)?;
        self.counts.payload_bytes = add(self.counts.payload_bytes, payload)?;
        self.counts.frames = add(self.counts.frames, 1)?;
        if version == VERSION_WIDE {
            self.counts.wide_frames = add(self.counts.wide_frames, 1)?;
        }
        if version == VERSION_WIDE256 {
            self.counts.wide256_frames = add(self.counts.wide256_frames, 1)?;
        }
        if version == VERSION_WIDE1024 {
            self.counts.wide1024_frames = add(self.counts.wide1024_frames, 1)?;
        }
        self.counts.total_bytes = self.written;
        Ok(())
    }

    fn answer(&mut self, kind: Kind, answer: &Symbol<F128>) -> io::Result<()> {
        let fields = len(answer.0.len())?;
        let payload = fields
            .checked_mul(ELEMENT_BYTES as u64)
            .and_then(|n| n.checked_add(8))
            .ok_or_else(|| invalid("wire answer length overflow"))?;
        self.frame(kind, payload, |out| {
            out.u64(fields)?;
            for field in &answer.0 {
                out.write_all(&field.to_be_bytes())?;
            }
            Ok(())
        })
    }
}

pub fn write_execution<W: Write>(
    sink: W,
    transcript: &Transcript,
    execution: &Execution,
    audited: &Occurrence,
    checker_answer: &Symbol<F128>,
) -> io::Result<CommunicationBytes> {
    write_execution_with_domains(sink, transcript, execution, audited, checker_answer, |_| {
        None
    })
}

pub fn write_execution_with_domains<W: Write>(
    sink: W,
    transcript: &Transcript,
    execution: &Execution,
    audited: &Occurrence,
    checker_answer: &Symbol<F128>,
    domain_of: impl Fn(OracleId) -> Option<Coord>,
) -> io::Result<CommunicationBytes> {
    if execution.transcript_at_query_start != execution.transcript_at_query_end
        || execution.transcript_at_query_end != transcript.entries().len()
        || execution.transcript_before_sampling > execution.transcript_at_query_start
    {
        return Err(invalid(
            "wire requires the completed deferred-query transcript",
        ));
    }
    if !execution.occurrences.iter().any(|occurrence| {
        occurrence.oracle == audited.oracle
            && occurrence.coord == audited.coord
            && occurrence.prefix == audited.prefix
            && occurrence.answer == audited.answer
    }) {
        return Err(invalid("wire audited occurrence is absent from execution"));
    }
    for occurrence in &execution.occurrences {
        if transcript.try_prefix_at(occurrence.oracle).as_ref() != Some(&occurrence.prefix) {
            return Err(invalid("wire occurrence has the wrong posting prefix"));
        }
    }
    let mut out = Encoder {
        sink,
        written: 0,
        counts: CommunicationBytes::default(),
    };
    for entry in transcript.entries() {
        match entry {
            Entry::Ordinary(bytes) => out.frame(Kind::MainOrdinary, len(bytes.len())?, |out| {
                out.write_all(bytes)
            })?,
            Entry::Coin(coin) => out.frame(Kind::PublicCoin, 16, |out| {
                out.write_all(&coin.to_be_bytes())
            })?,
            Entry::OraclePost(oracle) => {
                let kind = if *oracle == DUMMY_ORACLE {
                    Kind::PublicOraclePost
                } else {
                    Kind::MainOraclePost
                };
                out.frame(kind, 8, |out| out.u64(len(*oracle)?))?;
            }
        }
    }
    for occurrence in &execution.occurrences {
        let width = coord_width(domain_of(occurrence.oracle));
        let version = version_for(width);
        out.frame_at(version, Kind::MainQuery, 8 + u64::from(width), |out| {
            out.u64(len(occurrence.oracle)?)?;
            out.coord(width, occurrence.coord)
        })?;
        out.answer(Kind::MainAnswer, &occurrence.answer)?;
    }
    let prefix = audited.prefix.as_bytes();
    let width = coord_width(domain_of(audited.oracle));
    let version = version_for(width);
    out.frame_at(
        version,
        Kind::CheckerQuery,
        add(16 + u64::from(width), len(prefix.len())?)?,
        |out| {
            out.u64(len(audited.oracle)?)?;
            out.u64(len(prefix.len())?)?;
            out.write_all(prefix)?;
            out.coord(width, audited.coord)
        },
    )?;
    out.answer(Kind::CheckerAnswer, checker_answer)?;
    Ok(out.counts)
}

pub fn write_execution_sequential_with_domains<W: Write>(
    sink: W,
    transcript: &Transcript,
    execution: &Execution,
    answers: &[Symbol<F128>],
    domain_of: impl Fn(OracleId) -> Option<Coord>,
) -> io::Result<CommunicationBytes> {
    if execution.transcript_at_query_start != execution.transcript_at_query_end
        || execution.transcript_at_query_end != transcript.entries().len()
        || execution.transcript_before_sampling > execution.transcript_at_query_start
    {
        return Err(invalid(
            "wire requires the completed deferred-query transcript",
        ));
    }
    if answers.len() > execution.occurrences.len() {
        return Err(invalid("more checker answers than occurrences"));
    }
    for occurrence in &execution.occurrences {
        if transcript.try_prefix_at(occurrence.oracle).as_ref() != Some(&occurrence.prefix) {
            return Err(invalid("wire occurrence has the wrong posting prefix"));
        }
    }
    let mut out = Encoder {
        sink,
        written: 0,
        counts: CommunicationBytes::default(),
    };
    for entry in transcript.entries() {
        match entry {
            Entry::Ordinary(bytes) => out.frame(Kind::MainOrdinary, len(bytes.len())?, |out| {
                out.write_all(bytes)
            })?,
            Entry::Coin(coin) => out.frame(Kind::PublicCoin, 16, |out| {
                out.write_all(&coin.to_be_bytes())
            })?,
            Entry::OraclePost(oracle) => {
                let kind = if *oracle == DUMMY_ORACLE {
                    Kind::PublicOraclePost
                } else {
                    Kind::MainOraclePost
                };
                out.frame(kind, 8, |out| out.u64(len(*oracle)?))?;
            }
        }
    }
    for occurrence in &execution.occurrences {
        let width = coord_width(domain_of(occurrence.oracle));
        let version = version_for(width);
        out.frame_at(version, Kind::MainQuery, 8 + u64::from(width), |out| {
            out.u64(len(occurrence.oracle)?)?;
            out.coord(width, occurrence.coord)
        })?;
        out.answer(Kind::MainAnswer, &occurrence.answer)?;
    }
    for (occurrence, answer) in execution.occurrences.iter().zip(answers) {
        let prefix = occurrence.prefix.as_bytes();
        let width = coord_width(domain_of(occurrence.oracle));
        let version = version_for(width);
        out.frame_at(
            version,
            Kind::CheckerQuery,
            add(16 + u64::from(width), len(prefix.len())?)?,
            |out| {
                out.u64(len(occurrence.oracle)?)?;
                out.u64(len(prefix.len())?)?;
                out.write_all(prefix)?;
                out.coord(width, occurrence.coord)
            },
        )?;
        out.answer(Kind::CheckerAnswer, answer)?;
    }
    Ok(out.counts)
}

pub fn measure_execution_sequential_with_domains(
    transcript: &Transcript,
    execution: &Execution,
    answers: &[Symbol<F128>],
    domain_of: impl Fn(OracleId) -> Option<Coord>,
) -> io::Result<CommunicationBytes> {
    write_execution_sequential_with_domains(io::sink(), transcript, execution, answers, domain_of)
}

pub fn measure_execution(
    transcript: &Transcript,
    execution: &Execution,
    audited: &Occurrence,
    checker_answer: &Symbol<F128>,
) -> io::Result<CommunicationBytes> {
    write_execution(io::sink(), transcript, execution, audited, checker_answer)
}

pub fn measure_execution_with_domains(
    transcript: &Transcript,
    execution: &Execution,
    audited: &Occurrence,
    checker_answer: &Symbol<F128>,
    domain_of: impl Fn(OracleId) -> Option<Coord>,
) -> io::Result<CommunicationBytes> {
    write_execution_with_domains(
        io::sink(),
        transcript,
        execution,
        audited,
        checker_answer,
        domain_of,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame<'a> {
    pub kind: Kind,
    pub payload: &'a [u8],
    pub version: u8,
}

fn take<'a>(rest: &mut &'a [u8], n: usize) -> io::Result<&'a [u8]> {
    if rest.len() < n {
        return Err(invalid("truncated wire frame"));
    }
    let (head, tail) = rest.split_at(n);
    *rest = tail;
    Ok(head)
}

fn read_u64(rest: &mut &[u8]) -> io::Result<u64> {
    Ok(u64::from_be_bytes(take(rest, 8)?.try_into().unwrap()))
}

fn read_u128(rest: &mut &[u8]) -> io::Result<u128> {
    Ok(u128::from_be_bytes(take(rest, 16)?.try_into().unwrap()))
}

fn read_coord(rest: &mut &[u8], version: u8) -> io::Result<Coord> {
    if version == VERSION_WIDE1024 {
        Ok(Coord::from_be_bytes(take(rest, 128)?.try_into().unwrap()))
    } else if version == VERSION_WIDE256 {
        let mut wide = [0u8; 128];
        wide[96..].copy_from_slice(take(rest, 32)?);
        Ok(Coord::from_be_bytes(wide))
    } else if version == VERSION_WIDE {
        Ok(Coord::from(read_u128(rest)?))
    } else {
        Ok(Coord::from(read_u64(rest)?))
    }
}

fn read_len(rest: &mut &[u8]) -> io::Result<usize> {
    usize::try_from(read_u64(rest)?).map_err(|_| invalid("wire length exceeds usize"))
}

impl Frame<'_> {
    pub fn checker_question(&self) -> io::Result<(u64, TranscriptPrefix, Coord)> {
        if self.kind != Kind::CheckerQuery {
            return Err(invalid("wire frame is not a checker question"));
        }
        let mut rest = self.payload;
        let oracle = read_u64(&mut rest)?;
        let prefix_len = read_len(&mut rest)?;
        let prefix = TranscriptPrefix::from_bytes(take(&mut rest, prefix_len)?)
            .map_err(|_| invalid("wire checker prefix is malformed"))?;
        let coord = read_coord(&mut rest, self.version)?;
        if !rest.is_empty() {
            return Err(invalid("wire checker question has trailing data"));
        }
        Ok((oracle, prefix, coord))
    }
}

pub fn decode(mut bytes: &[u8]) -> io::Result<Vec<Frame<'_>>> {
    let mut frames = Vec::new();
    while !bytes.is_empty() {
        if take(&mut bytes, 4)? != MAGIC {
            return Err(invalid("wire magic mismatch"));
        }
        let version = take(&mut bytes, 1)?[0];
        if ![VERSION, VERSION_WIDE, VERSION_WIDE256, VERSION_WIDE1024].contains(&version) {
            return Err(invalid("unsupported wire version"));
        }
        let kind = match take(&mut bytes, 1)?[0] {
            1 => Kind::MainOrdinary,
            2 => Kind::PublicCoin,
            3 => Kind::MainOraclePost,
            4 => Kind::PublicOraclePost,
            5 => Kind::MainQuery,
            6 => Kind::MainAnswer,
            7 => Kind::CheckerQuery,
            8 => Kind::CheckerAnswer,
            _ => return Err(invalid("unknown wire message kind")),
        };
        let payload_len = read_len(&mut bytes)?;
        let payload = take(&mut bytes, payload_len)?;
        let frame = Frame {
            kind,
            payload,
            version,
        };
        let coordinate_frame = matches!(kind, Kind::MainQuery | Kind::CheckerQuery);
        if version != VERSION && !coordinate_frame {
            return Err(invalid("wide wire version on a frame without a coordinate"));
        }
        let valid = match kind {
            Kind::MainOrdinary => true,
            Kind::PublicCoin => payload.len() == 16,
            Kind::MainQuery => payload.len() == 8 + usize::from(coord_bytes(version)),
            Kind::MainOraclePost | Kind::PublicOraclePost => {
                if payload.len() != 8 {
                    false
                } else {
                    let oracle = read_u64(&mut &payload[..])?;
                    (kind == Kind::PublicOraclePost) == (oracle == len(DUMMY_ORACLE)?)
                }
            }
            Kind::MainAnswer | Kind::CheckerAnswer => {
                let mut rest = payload;
                let fields = read_u64(&mut rest)?;
                fields.checked_mul(16) == Some(len(rest.len())?)
            }
            Kind::CheckerQuery => {
                frame.checker_question()?;
                true
            }
        };
        if !valid {
            return Err(invalid("wire payload has a noncanonical length or value"));
        }
        frames.push(frame);
    }
    Ok(frames)
}
