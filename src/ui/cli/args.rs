use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueHint};
use serde_json::{Map, Value};

use crate::ui::types::choices::{
    DumpFormat, EvaluatorChoice, LearnerChoice, StreamChoice, TaskChoice, TaskKind, UIChoice,
};

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Interactive and scripted runner for rivu tasks"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a task without the interactive wizard
    Run(RunArgs),
}

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Task to execute (e.g. evaluate-prequential)
    #[arg(long, default_value = "evaluate-prequential", value_name = "TASK")]
    pub task: String,

    /// Learner to use (e.g. naive-bayes, HoeffdingTree)
    #[arg(long, value_name = "LEARNER")]
    pub learner: String,

    /// Stream to consume (e.g. sea-generator)
    #[arg(long, value_name = "STREAM")]
    pub stream: String,

    /// Evaluator to record metrics (e.g. basic-classification)
    #[arg(long, value_name = "EVALUATOR")]
    pub evaluator: String,

    /// Stop after this many instances (omit for unlimited)
    #[arg(long, value_name = "N")]
    pub max_instances: Option<u64>,

    /// Stop after this many seconds (omit for unlimited)
    #[arg(long, value_name = "SECONDS")]
    pub max_seconds: Option<u64>,

    /// Emit metrics every N instances
    #[arg(
        long,
        default_value_t = 100_000,
        value_name = "N",
        value_parser = clap::value_parser!(u64).range(1..),
    )]
    pub sample_frequency: u64,

    /// Check memory usage every N instances
    #[arg(
        long,
        default_value_t = 100_000,
        value_name = "N",
        value_parser = clap::value_parser!(u64).range(1..),
    )]
    pub mem_check_frequency: u64,

    /// File to dump evaluation snapshots after completion
    #[arg(long, value_name = "PATH", value_hint = ValueHint::FilePath)]
    pub dump_file: Option<PathBuf>,

    /// Format for the dump file (csv, tsv, json)
    #[arg(long, value_name = "FORMAT")]
    pub dump_format: Option<String>,

    /// Override learner parameters (key=value, nested keys with dots)
    #[arg(long = "learner-param", value_name = "KEY=VALUE", value_parser = parse_key_value)]
    pub learner_params: Vec<KeyValue>,

    /// Override stream parameters (key=value, nested keys with dots)
    #[arg(long = "stream-param", value_name = "KEY=VALUE", value_parser = parse_key_value)]
    pub stream_params: Vec<KeyValue>,

    /// Override evaluator parameters (key=value, nested keys with dots)
    #[arg(
        long = "evaluator-param",
        value_name = "KEY=VALUE",
        value_parser = parse_key_value
    )]
    pub evaluator_params: Vec<KeyValue>,
}

#[derive(Clone, Debug)]
pub struct KeyValue {
    key: String,
    value: Value,
}

impl RunArgs {
    pub fn into_task_choice(self) -> Result<TaskChoice> {
        let task_kind = parse_kind::<TaskKind>(&self.task)
            .with_context(|| format!("invalid task '{}'", self.task))?;

        match task_kind {
            TaskKind::EvaluatePrequential => self.into_prequential_choice(),
        }
    }

    fn into_prequential_choice(self) -> Result<TaskChoice> {
        let learner_choice = build_choice::<LearnerChoice>(&self.learner, &self.learner_params)
            .with_context(|| format!("invalid learner '{}'", self.learner))?;
        let stream_choice = build_choice::<StreamChoice>(&self.stream, &self.stream_params)
            .with_context(|| format!("invalid stream '{}'", self.stream))?;
        let evaluator_choice =
            build_choice::<EvaluatorChoice>(&self.evaluator, &self.evaluator_params)
                .with_context(|| format!("invalid evaluator '{}'", self.evaluator))?;

        let dump_format = match self.dump_format {
            Some(fmt) => Some(
                parse_dump_format(&fmt).with_context(|| format!("invalid dump format '{fmt}'"))?,
            ),
            None => None,
        };

        let params = crate::ui::types::choices::PrequentialParams {
            learner: learner_choice,
            stream: stream_choice,
            evaluator: evaluator_choice,
            max_instances: self.max_instances,
            max_seconds: self.max_seconds,
            sample_frequency: self.sample_frequency,
            mem_check_frequency: self.mem_check_frequency,
            dump_file: self.dump_file,
            dump_format: dump_format.unwrap_or_default(),
        };

        Ok(TaskChoice::EvaluatePrequential(params))
    }
}

fn build_choice<C>(kind_input: &str, overrides: &[KeyValue]) -> Result<C>
where
    C: UIChoice,
    C::Kind: FromStr,
    <C::Kind as FromStr>::Err: std::fmt::Display,
{
    let kind = parse_kind::<C::Kind>(kind_input)?;
    let mut params = C::default_params(kind);
    apply_overrides(&mut params, overrides)?;
    C::from_parts(kind, params)
}

fn parse_kind<T>(raw: &str) -> Result<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
{
    let candidates = candidate_spellings(raw);
    for cand in candidates {
        if let Ok(parsed) = cand.parse::<T>() {
            return Ok(parsed);
        }
    }
    Err(anyhow!("could not parse value '{raw}'"))
}

fn candidate_spellings(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return out;
    }

    out.push(trimmed.to_string());
    out.push(trimmed.to_lowercase());
    out.push(kebab_from_token(trimmed));
    out.push(trimmed.replace('_', "-"));
    out.sort();
    out.dedup();
    out
}

fn kebab_from_token(token: &str) -> String {
    let mut buf = String::new();
    let mut prev_lower = false;
    for ch in token.chars() {
        if ch.is_uppercase() {
            if prev_lower {
                buf.push('-');
            }
            for low in ch.to_lowercase() {
                buf.push(low);
            }
            prev_lower = false;
        } else {
            if ch == '_' {
                buf.push('-');
                prev_lower = false;
            } else {
                buf.push(ch);
                prev_lower = ch.is_lowercase();
            }
        }
    }
    if buf.is_empty() {
        token.to_lowercase()
    } else {
        buf
    }
}

fn parse_dump_format(input: &str) -> Result<DumpFormat> {
    match input.trim().to_lowercase().as_str() {
        "csv" => Ok(DumpFormat::Csv),
        "tsv" => Ok(DumpFormat::Tsv),
        "json" => Ok(DumpFormat::Json),
        other => Err(anyhow!("unknown format '{other}'")),
    }
}

fn apply_overrides(target: &mut Value, overrides: &[KeyValue]) -> Result<()> {
    for kv in overrides {
        set_path(target, &kv.key, kv.value.clone())
            .with_context(|| format!("failed to set '{}'", kv.key))?;
    }
    Ok(())
}

fn set_path(target: &mut Value, path: &str, new_value: Value) -> Result<()> {
    let segments: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        bail!("empty key is not allowed");
    }

    let mut current = target;
    for seg in &segments[..segments.len() - 1] {
        ensure_object(current)?;
        current = current
            .as_object_mut()
            .expect("object after ensure")
            .entry((*seg).to_string())
            .or_insert(Value::Null);
    }

    ensure_object(current)?;
    current
        .as_object_mut()
        .expect("object after ensure")
        .insert(segments.last().unwrap().to_string(), new_value);
    Ok(())
}

fn ensure_object(value: &mut Value) -> Result<()> {
    match value {
        Value::Object(_) => Ok(()),
        Value::Null => {
            *value = Value::Object(Map::new());
            Ok(())
        }
        other => bail!("cannot set nested field on non-object value: {other:?}"),
    }
}

fn parse_key_value(raw: &str) -> Result<KeyValue, String> {
    let (key, value) = raw
        .split_once('=')
        .ok_or_else(|| "expected KEY=VALUE".to_string())?;
    let key = key.trim();
    if key.is_empty() {
        return Err("key cannot be empty".to_string());
    }

    let value = value.trim();
    let parsed = parse_literal(value).map_err(|e| e.to_string())?;

    Ok(KeyValue {
        key: key.to_string(),
        value: parsed,
    })
}

fn parse_literal(raw: &str) -> Result<Value> {
    if raw.is_empty() {
        return Ok(Value::String(String::new()));
    }

    match serde_json::from_str(raw) {
        Ok(v) => Ok(v),
        Err(_) => Ok(Value::String(raw.to_string())),
    }
}
