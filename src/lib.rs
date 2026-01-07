pub mod allocator;
pub mod config;
pub mod handlers;
pub mod processors;
pub mod utils;

pub const FN_TAGS: &str = "tags";
pub const FN_CLITICS: &str = "clitics";
pub const FN_TAGS_FREQ: &str = "tags_freq";

pub const MI_URL: &str = "Dl";
pub const MI_EMAIL: &str = "De";

const SYMBOLS: &str =
    ".!?;:#$£€¢¥₹%&()[]{}*,–—-/\\<>≤≥=@^_|`~´¦…„“’¬●¨■▪◾•◦§¶°′″«»‹›®™©℗−+±×÷→←↑↓↕↔↖↗↘↙✓✔†‡";
const MATH_SYMBOLS: &str = "¹²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞";
// const GREEK_LETTERS: &str = "αβΓγΔδεζηΘθικΛλμνΞξπρΣστυΦφχΨψΩω";
