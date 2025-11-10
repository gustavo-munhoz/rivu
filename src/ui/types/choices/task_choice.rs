use crate::evaluation::CurveFormat;
use crate::ui::cli::wizard::prompt_choice;
use crate::ui::types::choices::{EvaluatorChoice, LearnerChoice, StreamChoice, UIChoice};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::path::PathBuf;
use strum_macros::{Display, EnumDiscriminants, EnumIter, EnumMessage, EnumString, IntoStaticStr};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum DumpFormat {
    Csv,
    Tsv,
    Json,
}

impl Default for DumpFormat {
    fn default() -> Self {
        DumpFormat::Csv
    }
}

impl From<DumpFormat> for CurveFormat {
    fn from(value: DumpFormat) -> Self {
        match value {
            DumpFormat::Csv => CurveFormat::Csv,
            DumpFormat::Tsv => CurveFormat::Tsv,
            DumpFormat::Json => CurveFormat::Json,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PrequentialParams {
    #[schemars(skip)]
    pub learner: LearnerChoice,
    #[schemars(skip)]
    pub stream: StreamChoice,
    #[schemars(skip)]
    pub evaluator: EvaluatorChoice,

    #[serde(default)]
    #[schemars(
        title = "Max Instances",
        description = "Stop after this many instances (None = unlimited)"
    )]
    pub max_instances: Option<u64>,

    #[serde(default)]
    #[schemars(
        title = "Max Seconds",
        description = "Stop after this many seconds (None = unlimited)"
    )]
    pub max_seconds: Option<u64>,

    #[schemars(
        title = "Sample Frequency",
        description = "Emit metrics every N instances",
        range(min = 1)
    )]
    pub sample_frequency: u64,

    #[schemars(
        title = "Memory Check Frequency",
        description = "Check memory every N instances",
        range(min = 1)
    )]
    pub mem_check_frequency: u64,

    #[serde(default)]
    #[schemars(
        with = "String",
        title = "Dump file",
        description = "If set, write all snapshots at the end to this file",
        extend("format"="path","x-file"=true,"x-must-exist"=false)
    )]
    pub dump_file: Option<PathBuf>,

    #[serde(default)]
    #[schemars(title = "Dump format", description = "csv / tsv / json (default: csv)")]
    pub dump_format: DumpFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, EnumDiscriminants)]
#[serde(tag = "type", content = "params", rename_all = "kebab-case")]
#[strum_discriminants(name(TaskKind))]
#[strum_discriminants(derive(EnumIter, EnumString, Display, IntoStaticStr, EnumMessage))]
#[strum_discriminants(strum(serialize_all = "kebab-case"))]
pub enum TaskChoice {
    #[strum_discriminants(strum(
        message = "Evaluate Prequential",
        detailed_message = "Interleave test-then-train with periodic reporting."
    ))]
    EvaluatePrequential(PrequentialParams),
}

impl UIChoice for TaskChoice {
    type Kind = TaskKind;

    fn schema() -> Schema {
        schema_for!(TaskChoice)
    }

    fn prompt_label() -> &'static str {
        "Choose a task:"
    }
    fn default_params(kind: Self::Kind) -> Value {
        match kind {
            TaskKind::EvaluatePrequential => json!({
                "max_instances": null,
                "max_seconds": null,
                "sample_frequency": 100_000,
                "mem_check_frequency": 100_000,
                "dump_file": null,
                "dump_format": "csv"
            }),
        }
    }

    fn subprompts<D: crate::ui::cli::drivers::PromptDriver>(
        driver: &D,
        kind: Self::Kind,
    ) -> anyhow::Result<Option<Map<String, Value>>> {
        match kind {
            TaskKind::EvaluatePrequential => {
                let learner = prompt_choice::<LearnerChoice, _>(driver)?;
                let stream = prompt_choice::<StreamChoice, _>(driver)?;
                let eval = prompt_choice::<EvaluatorChoice, _>(driver)?;

                let mut m = Map::new();
                m.insert("learner".into(), serde_json::to_value(learner)?);
                m.insert("stream".into(), serde_json::to_value(stream)?);
                m.insert("evaluator".into(), serde_json::to_value(eval)?);
                Ok(Some(m))
            }
        }
    }

    fn from_parts(kind: Self::Kind, params: Value) -> anyhow::Result<Self> {
        match kind {
            TaskKind::EvaluatePrequential => {
                let p: PrequentialParams = serde_json::from_value(params)?;
                Ok(TaskChoice::EvaluatePrequential(p))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::types::choices::{
        EvaluatorChoice, EvaluatorKind, LearnerChoice, LearnerKind, StreamChoice, StreamKind,
        UIChoice,
    };
    use schemars::schema_for;
    use serde_json::{Value, json};

    fn root_schema_json<T: JsonSchema>() -> Value {
        serde_json::to_value(schema_for!(T)).expect("schema to JSON")
    }
    fn root_props_of<T: JsonSchema>() -> Value {
        let v = root_schema_json::<T>();
        v.get("schema")
            .cloned()
            .unwrap_or(v)
            .get("properties")
            .cloned()
            .unwrap_or_else(|| json!({}))
    }

    fn make_choice_json<C: UIChoice>(kind: C::Kind) -> Value {
        let params = <C as UIChoice>::default_params(kind);
        let choice = <C as UIChoice>::from_parts(kind, params).expect("from_parts");
        serde_json::to_value(choice).expect("choice -> json")
    }

    #[test]
    fn default_params_have_expected_sampling_values() {
        let v = <TaskChoice as UIChoice>::default_params(TaskKind::EvaluatePrequential);
        let obj = v.as_object().expect("params object");
        assert_eq!(obj.get("max_instances").unwrap(), &Value::Null);
        assert_eq!(obj.get("max_seconds").unwrap(), &Value::Null);

        assert_eq!(
            obj.get("sample_frequency").and_then(Value::as_u64),
            Some(100_000)
        );
        assert_eq!(
            obj.get("mem_check_frequency").and_then(Value::as_u64),
            Some(100_000)
        );
    }

    #[test]
    fn from_parts_builds_prequential_with_nested_choices() {
        let learner_json = make_choice_json::<LearnerChoice>(LearnerKind::NaiveBayes);
        let stream_json = make_choice_json::<StreamChoice>(StreamKind::SeaGenerator);
        let evaluator_json =
            make_choice_json::<EvaluatorChoice>(EvaluatorKind::BasicClassification);

        let params = json!({
            "learner": learner_json,
            "stream":  stream_json,
            "evaluator": evaluator_json,
            "max_instances": 123u64,
            "max_seconds": null,
            "sample_frequency": 10u64,
            "mem_check_frequency": 50u64,
        });

        let tc = <TaskChoice as UIChoice>::from_parts(TaskKind::EvaluatePrequential, params)
            .expect("TaskChoice::from_parts");

        match tc {
            TaskChoice::EvaluatePrequential(p) => {
                assert_eq!(p.max_instances, Some(123));
                assert_eq!(p.max_seconds, None);
                assert_eq!(p.sample_frequency, 10);
                assert_eq!(p.mem_check_frequency, 50);

                let l = serde_json::to_value(&p.learner).unwrap();
                assert_eq!(l.get("type").and_then(Value::as_str), Some("naive-bayes"));

                let s = serde_json::to_value(&p.stream).unwrap();
                assert!(s.get("type").is_some());
                assert_eq!(s.get("type").and_then(Value::as_str), Some("sea-generator"));

                let e = serde_json::to_value(&p.evaluator).unwrap();
                assert_eq!(
                    e.get("type").and_then(Value::as_str),
                    Some("basic-classification")
                );
            }
        }
    }

    #[test]
    fn taskchoice_serializes_as_tagged_enum() {
        let learner_json = make_choice_json::<LearnerChoice>(LearnerKind::NaiveBayes);
        let stream_json = make_choice_json::<StreamChoice>(StreamKind::SeaGenerator);
        let evaluator_json =
            make_choice_json::<EvaluatorChoice>(EvaluatorKind::BasicClassification);

        let p = PrequentialParams {
            learner: serde_json::from_value(learner_json).unwrap(),
            stream: serde_json::from_value(stream_json).unwrap(),
            evaluator: serde_json::from_value(evaluator_json).unwrap(),
            max_instances: None,
            max_seconds: None,
            sample_frequency: 1000,
            mem_check_frequency: 1000,
            dump_file: None,
            dump_format: DumpFormat::Csv,
        };

        let v = serde_json::to_value(TaskChoice::EvaluatePrequential(p)).unwrap();
        assert_eq!(
            v.get("type").and_then(Value::as_str),
            Some("evaluate-prequential")
        );
        let params = v
            .get("params")
            .and_then(Value::as_object)
            .expect("params object");
        for k in [
            "sample_frequency",
            "mem_check_frequency",
            "max_instances",
            "max_seconds",
            "learner",
            "stream",
            "evaluator",
        ] {
            assert!(params.contains_key(k), "missing {k} in params");
        }
    }

    #[test]
    fn prequential_params_schema_has_ranges_and_no_skipped_fields() {
        let props = root_props_of::<PrequentialParams>();
        let obj = props.as_object().unwrap();

        assert!(!obj.contains_key("learner"));
        assert!(!obj.contains_key("stream"));
        assert!(!obj.contains_key("evaluator"));

        let sf = obj.get("sample_frequency").unwrap().as_object().unwrap();
        assert_eq!(sf.get("minimum").and_then(Value::as_u64), Some(1));
        assert_eq!(
            sf.get("title").and_then(Value::as_str),
            Some("Sample Frequency")
        );

        let mf = obj.get("mem_check_frequency").unwrap().as_object().unwrap();
        assert_eq!(mf.get("minimum").and_then(Value::as_u64), Some(1));
        assert_eq!(
            mf.get("title").and_then(Value::as_str),
            Some("Memory Check Frequency")
        );

        assert!(obj.contains_key("max_instances"));
        assert!(obj.contains_key("max_seconds"));
    }

    #[test]
    fn prompt_label_is_expected() {
        assert_eq!(<TaskChoice as UIChoice>::prompt_label(), "Choose a task:");
    }
}
