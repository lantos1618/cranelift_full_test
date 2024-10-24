#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    IntLiteral(i64),
    FloatLiteral(f64),
    BoolLiteral(bool),
    StringLiteral(String),
    CharLiteral(char),
    Identifier(String),
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    Equal,       // =
    NotEqual,    // !=
    LessThan,    // <
    GreaterThan, // >
    LessEqual,   // <=
    GreaterEqual,// >=
    And,         // &&
    Or,          // ||
    Not,         // !
    BitAnd,      // &
    BitOr,       // |
    BitXor,      // ^
    ShiftLeft,   // <<
    ShiftRight,  // >>
    OpenBrace,   // {
    CloseBrace,  // }
    OpenParen,   // (
    CloseParen,  // )
    Comma,       // ,
    Colon,       // :
    Arrow,       // =>
    Do,
    Match,
    If,
    Else,
    While,
    For,
    Return,
    Loop,
    // Add any other necessary tokens here
}
