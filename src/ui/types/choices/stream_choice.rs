use crate::ui::types::choices::UIChoice;
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use strum_macros::{Display, EnumDiscriminants, EnumIter, EnumMessage, EnumString, IntoStaticStr};

const DEFAULT_SEED: u64 = 42;
fn default_seed() -> u64 {
    DEFAULT_SEED
}

fn default_sea_function() -> u8 {
    2
}

fn default_agrawal_function() -> u8 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct ArffParameters {
    #[schemars(
        with = "String",
        title = "ARFF Path",
        description = "Path to .arff file",
        extend(
            "format" = "path",
            "x-file" = true,
            "x-must-exist" = true,
            "x-extensions" = ["arff"]
        )
    )]
    pub path: PathBuf,

    #[schemars(
        title = "Class Index",
        description = "Index of the class column. (None = last attribute in file)",
        range(min = 1)
    )]
    pub class_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SeaParameters {
    #[serde(default = "default_sea_function")]
    #[schemars(
        title = "Function",
        description = "Classification SEA Function used (1-4)",
        range(min = 1, max = 4),
        default = "default_sea_function"
    )]
    pub function_id: u8,

    #[schemars(title = "Balance", description = "Balance classes during generation?")]
    pub balance: bool,

    #[schemars(
        title = "Noise",
        description = "Noise percentage (0.0–1.0)",
        range(min = 0.0, max = 1.0)
    )]
    pub noise_pct: f32,

    #[serde(default)]
    #[schemars(
        title = "Concept Instances Number",
        description = "The number of instances for each concept"
    )]
    pub max_instances: Option<u64>,

    #[serde(default = "default_seed")]
    #[schemars(title = "Seed", description = "PRNG seed", default = "default_seed")]
    pub seed: u64,
}

impl Default for SeaParameters {
    fn default() -> Self {
        Self {
            function_id: default_sea_function(),
            balance: false,
            noise_pct: 0.0,
            max_instances: None,
            seed: DEFAULT_SEED,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq)]
pub struct AgrawalParameters {
    #[schemars(
        title = "Function",
        description = "Agrawal function (1–10)",
        range(min = 1, max = 10),
        default = "default_agrawal_function"
    )]
    pub function_id: u8,

    #[schemars(title = "Balance", description = "Balance classes during generation?")]
    pub balance: bool,

    #[schemars(
        title = "Perturbation Fraction",
        description = "Drift/perturbation fraction (0.0–1.0)",
        range(min = 0.0, max = 1.0)
    )]
    pub perturb_fraction: f64,

    #[serde(default)]
    #[schemars(
        title = "Max Instances",
        description = "Upper bound on instances; empty = infinite"
    )]
    pub max_instances: Option<u64>,

    #[serde(default = "default_seed")]
    #[schemars(title = "Seed", description = "PRNG seed", default = "default_seed")]
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq)]
pub struct AssetNegotiationParameters {
    #[schemars(
        title = "Rule",
        description = "Concept rule (1-5)",
        range(min = 1, max = 5)
    )]
    pub rule_id: u8,

    #[schemars(title = "Balance", description = "Balance classes during generation?")]
    pub balance: bool,

    #[schemars(
        title = "Noise (%)",
        description = "Noise fraction (0.0–1.0)",
        range(min = 0.0, max = 1.0)
    )]
    pub noise_pct: f32,

    #[serde(default = "default_seed")]
    #[schemars(title = "Seed", description = "PRNG seed", default = "default_seed")]
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, EnumDiscriminants)]
#[serde(tag = "type", content = "params", rename_all = "kebab-case")]
#[strum_discriminants(name(StreamKind))]
#[strum_discriminants(derive(EnumIter, EnumString, Display, IntoStaticStr, EnumMessage))]
#[strum_discriminants(strum(serialize_all = "kebab-case"))]
pub enum StreamChoice {
    #[strum_discriminants(strum(
        message = "Arff File Stream",
        detailed_message = "A stream read from an ARFF file."
    ))]
    ArffFile(ArffParameters),

    #[strum_discriminants(strum(
        message = "SEA Generator",
        detailed_message = "Generates SEA concept functions."
    ))]
    SeaGenerator(SeaParameters),

    #[strum_discriminants(strum(
        message = "Agrawal Generator",
        detailed_message = "Generates one of ten different pre-defined loan functions."
    ))]
    AgrawalGenerator(AgrawalParameters),

    #[strum_discriminants(strum(
        message = "Asset Negotiation Generator",
        detailed_message = "Generates instances using 5 concept functions to model agent interest."
    ))]
    AssetNegotiationGenerator(AssetNegotiationParameters),
}

impl UIChoice for StreamChoice {
    type Kind = StreamKind;

    fn schema() -> Schema {
        schema_for!(StreamChoice)
    }

    fn prompt_label() -> &'static str {
        "Choose a stream:"
    }

    fn default_params(kind: Self::Kind) -> Value {
        match kind {
            StreamKind::ArffFile => serde_json::to_value(ArffParameters::default()).unwrap(),
            StreamKind::SeaGenerator => serde_json::to_value(SeaParameters::default()).unwrap(),
            StreamKind::AgrawalGenerator => {
                serde_json::to_value(AgrawalParameters::default()).unwrap()
            }
            StreamKind::AssetNegotiationGenerator => {
                serde_json::to_value(AssetNegotiationParameters::default()).unwrap()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::types::choices::UIChoice;
    use schemars::schema_for;
    use serde_json::{Value, json};
    use strum::EnumMessage;

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

    #[test]
    fn serde_roundtrip_arff() {
        let p0 = ArffParameters {
            path: PathBuf::from("data/a.arff"),
            class_index: Some(1),
        };
        let j = serde_json::to_string(&p0).unwrap();
        let p1: ArffParameters = serde_json::from_str(&j).unwrap();
        assert_eq!(p0.path, p1.path);
        assert_eq!(p0.class_index, p1.class_index);
    }

    #[test]
    fn serde_roundtrip_sea() {
        let p0 = SeaParameters {
            function_id: 2,
            balance: true,
            noise_pct: 0.25,
            max_instances: Some(123),
            seed: 42,
        };
        let j = serde_json::to_string(&p0).unwrap();
        let p1: SeaParameters = serde_json::from_str(&j).unwrap();
        assert_eq!(p0, p1);
    }

    #[test]
    fn serde_roundtrip_agrawal() {
        let p0 = AgrawalParameters {
            function_id: 7,
            balance: false,
            perturb_fraction: 0.1,
            max_instances: None,
            seed: 99,
        };
        let j = serde_json::to_string(&p0).unwrap();
        let p1: AgrawalParameters = serde_json::from_str(&j).unwrap();
        assert_eq!(p0, p1);
    }

    #[test]
    fn serde_roundtrip_asset() {
        let p0 = AssetNegotiationParameters {
            rule_id: 3,
            balance: true,
            noise_pct: 0.75,
            seed: 7,
        };
        let j = serde_json::to_string(&p0).unwrap();
        let p1: AssetNegotiationParameters = serde_json::from_str(&j).unwrap();
        assert_eq!(p0, p1);
    }

    #[test]
    fn default_params_match_struct_defaults_for_streams() {
        let sea_defaults = <StreamChoice as UIChoice>::default_params(StreamKind::SeaGenerator);
        let sea: SeaParameters = serde_json::from_value(sea_defaults).unwrap();
        assert_eq!(sea.function_id, 2);
        assert_eq!(sea.balance, false);
        assert_eq!(sea.noise_pct, 0.0);
        assert_eq!(sea.max_instances, None);
        assert_eq!(sea.seed, DEFAULT_SEED);

        let _arff: ArffParameters = serde_json::from_value(
            <StreamChoice as UIChoice>::default_params(StreamKind::ArffFile),
        )
        .unwrap();
        let _agr: AgrawalParameters = serde_json::from_value(
            <StreamChoice as UIChoice>::default_params(StreamKind::AgrawalGenerator),
        )
        .unwrap();
        let _asst: AssetNegotiationParameters = serde_json::from_value(
            <StreamChoice as UIChoice>::default_params(StreamKind::AssetNegotiationGenerator),
        )
        .unwrap();
    }

    #[test]
    fn from_parts_rebuilds_enum() {
        let sea_params = serde_json::to_value(SeaParameters::default()).unwrap();
        let e =
            <StreamChoice as UIChoice>::from_parts(StreamKind::SeaGenerator, sea_params).unwrap();
        matches!(e, StreamChoice::SeaGenerator(_));

        let arff_params = serde_json::to_value(ArffParameters::default()).unwrap();
        let e = <StreamChoice as UIChoice>::from_parts(StreamKind::ArffFile, arff_params).unwrap();
        matches!(e, StreamChoice::ArffFile(_));
    }

    #[test]
    fn tagged_enum_serialization_stream_choice() {
        let sea = StreamChoice::SeaGenerator(SeaParameters::default());
        let v = serde_json::to_value(sea).unwrap();
        assert_eq!(v.get("type").and_then(Value::as_str), Some("sea-generator"));
        assert!(v.get("params").is_some());

        let agr = StreamChoice::AgrawalGenerator(AgrawalParameters::default());
        let v2 = serde_json::to_value(agr).unwrap();
        assert_eq!(
            v2.get("type").and_then(Value::as_str),
            Some("agrawal-generator")
        );

        let arff = StreamChoice::ArffFile(ArffParameters::default());
        let v3 = serde_json::to_value(arff).unwrap();
        assert_eq!(v3.get("type").and_then(Value::as_str), Some("arff-file"));
    }

    #[test]
    fn arff_schema_path_has_vendor_extensions() {
        let props = root_props_of::<ArffParameters>();
        let obj = props.as_object().unwrap();
        let path = obj.get("path").unwrap().as_object().unwrap();

        assert_eq!(path.get("type").and_then(Value::as_str), Some("string"));
        assert_eq!(path.get("format").and_then(Value::as_str), Some("path"));
        assert_eq!(path.get("x-file").and_then(Value::as_bool), Some(true));

        let exts = path.get("x-extensions").and_then(Value::as_array).unwrap();
        assert!(exts.iter().any(|v| v.as_str() == Some("arff")));

        let cls = obj.get("class_index").unwrap().as_object().unwrap();
        assert_eq!(cls.get("minimum").and_then(Value::as_u64), Some(1));
    }

    #[test]
    fn sea_schema_has_ranges_and_defaults() {
        let props = root_props_of::<SeaParameters>();
        let obj = props.as_object().unwrap();

        let fid = obj.get("function_id").unwrap().as_object().unwrap();
        assert_eq!(fid.get("default").and_then(Value::as_u64), Some(2));
        assert_eq!(fid.get("minimum").and_then(Value::as_u64), Some(1));
        assert_eq!(fid.get("maximum").and_then(Value::as_u64), Some(4));

        let noise = obj.get("noise_pct").unwrap().as_object().unwrap();
        assert_eq!(noise.get("minimum").and_then(Value::as_f64), Some(0.0));
        assert_eq!(noise.get("maximum").and_then(Value::as_f64), Some(1.0));
    }

    #[test]
    fn agrawal_schema_has_ranges_and_declared_default() {
        let props = root_props_of::<AgrawalParameters>();
        let obj = props.as_object().unwrap();

        let fid = obj.get("function_id").unwrap().as_object().unwrap();
        assert_eq!(fid.get("default").and_then(Value::as_u64), Some(1));
        assert_eq!(fid.get("minimum").and_then(Value::as_u64), Some(1));
        assert_eq!(fid.get("maximum").and_then(Value::as_u64), Some(10));

        let pf = obj.get("perturb_fraction").unwrap().as_object().unwrap();
        assert_eq!(pf.get("minimum").and_then(Value::as_f64), Some(0.0));
        assert_eq!(pf.get("maximum").and_then(Value::as_f64), Some(1.0));
    }

    #[test]
    fn asset_schema_rules_and_noise_ranges() {
        let props = root_props_of::<AssetNegotiationParameters>();
        let obj = props.as_object().unwrap();

        let rid = obj.get("rule_id").unwrap().as_object().unwrap();
        assert_eq!(rid.get("minimum").and_then(Value::as_u64), Some(1));
        assert_eq!(rid.get("maximum").and_then(Value::as_u64), Some(5));

        let nz = obj.get("noise_pct").unwrap().as_object().unwrap();
        assert_eq!(nz.get("minimum").and_then(Value::as_f64), Some(0.0));
        assert_eq!(nz.get("maximum").and_then(Value::as_f64), Some(1.0));
    }

    #[test]
    fn streamkind_messages_exist() {
        assert_eq!(StreamKind::ArffFile.get_message(), Some("Arff File Stream"));
        assert_eq!(
            StreamKind::SeaGenerator.get_message(),
            Some("SEA Generator")
        );
        assert!(
            StreamKind::AgrawalGenerator
                .get_detailed_message()
                .is_some()
        );
    }

    #[test]
    fn prompt_label_is_expected() {
        assert_eq!(
            <StreamChoice as UIChoice>::prompt_label(),
            "Choose a stream:"
        );
    }
}
