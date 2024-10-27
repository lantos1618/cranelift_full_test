#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub column: usize,
    pub lexeme: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Fn,
    Let,
    If,
    Else,
    Match,
    Loop,
    Return,
    Struct,
    None,
    Some,
    Option,
    Mut,

    // Literals
    IntLiteral(i32),
    FloatLiteral(f64),
    StringLiteral(String),
    CharLiteral(char),
    BoolLiteral(bool),
    Identifier(String),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Equal,
    Arrow,      // ->
    FatArrow,   // =>
    BitAnd,
    BitOr,
    BitXor,
    Not,
    ShiftLeft,  // <<
    ShiftRight, // >>
    LessThan,
    GreaterThan,

    // Delimiters
    OpenParen,    // (
    CloseParen,   // )
    OpenBrace,    // {
    CloseBrace,   // }
    OpenBracket,  // [
    CloseBracket, // ]
    Comma,
    Colon,
    Semicolon,
    Dot,
}

impl Token {
    pub fn new(kind: TokenKind, line: usize, column: usize, lexeme: String) -> Self {
        Token {
            kind,
            line,
            column,
            lexeme,
        }
    }
}
