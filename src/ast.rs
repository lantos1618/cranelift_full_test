// ast.rs

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    IntLiteral(i64),
    BoolLiteral(bool),
    StringLiteral(String),
    CharLiteral(char),

    // Variable handling
    Variable(String),
    Assignment(Box<Expr>, Box<Expr>),

    // Unary Operations
    UnaryOp(UnaryOp, Box<Expr>),

    // Binary Operations
    BinaryOp(Box<Expr>, BinOp, Box<Expr>),

    // Function calls
    FuncCall(String, Vec<Expr>),

    // Struct Access
    StructAccess(Box<Expr>, String),

    // Struct instantiation
    StructInit {
        struct_name: String,
        fields: Vec<(String, Expr)>,
    },

    // Extern calls
    ExternCall {
        func_name: String,
        args: Vec<Expr>,
        return_type: AstType,
    },

    // Match expression
    Match {
        expression: Box<Expr>,
        arms: Vec<(Pattern, Stmt)>,
    },

    // Array access and assignment
    ArrayAccess {
        array: Box<Expr>,
        index: Box<Expr>,
    },
    ArrayAssignment {
        array: Box<Expr>,
        index: Box<Expr>,
        value: Box<Expr>,
    },
}

// Binary Operations
#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add,         // +
    Subtract,    // -
    Multiply,    // *
    Divide,      // /
    Modulus,     // %   // Added modulus operator
    And,         // &&
    Or,          // ||
    Equal,       // ==
    NotEqual,    // !=
    LessThan,    // <
    GreaterThan, // >
    BitAnd,      // &
    BitOr,       // |
    BitXor,      // ^
    ShiftLeft,   // <<
    ShiftRight,  // >>
}

// Unary Operations
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Negate,    // - (numeric negation)
    Not,       // ! (logical negation)
    Deref,     // * (dereference a pointer)
    AddressOf, // & (get the address of a variable)
    BitNot,    // ~ (bitwise NOT)
}

// Pattern Matching
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Literal(Expr), // Matching literals (e.g., 42)
    Variable(String), // Variable binding (e.g., x)
    Wildcard, // Wildcard (_) pattern
    StructPattern {
        struct_name: String,
        fields: Vec<(String, Pattern)>,
    },
}

// Function declaration with types for parameters
#[derive(Debug, Clone, PartialEq    )]
pub struct FuncDecl {
    pub name: String,
    pub params: Vec<(String, AstType)>, // Name and type of parameters
    pub return_type: AstType,
}

// Statement (Stmt) as before with minor additions
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    // Variable declaration
    VarDecl {
        name: String,
        var_type: AstType,
        init_expr: Option<Box<Expr>>,
    },

    VarAssign {
        name: String,
        expr: Box<Expr>,
    },

    // Block of statements
    Block(Vec<Stmt>),

    // Expression as a statement
    ExprStmt(Box<Expr>),

    // Return statement
    Return(Box<Expr>),

    // Function definition
    FuncDef {
        func_decl: FuncDecl, // Function declaration with params and return type
        body: Box<Stmt>,     // Function body (a block of statements)
    },

    // Loop related statements
    Break,
    Continue,

    // External function declaration
    FuncExternDecl {
        name: FuncDecl,
        lib: String,
    },

    // Struct definition
    StructDef {
        name: String,
        fields: Vec<(String, AstType)>,
    },

    // Conditional statement
    If {
        condition: Box<Expr>,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },

    // Loop statement
    While {
        condition: Box<Expr>,
        body: Box<Stmt>,
    },
}

// Type system with Pointers, Generics, and Aliases
#[derive(Debug, Clone, PartialEq)]
pub enum AstType {
    Int,
    Bool,
    String,
    Char,
    Array(Box<AstType>),
    Struct(String),
    Void,

    // Pointer type (e.g., *int)
    Pointer(Box<AstType>),

    // Generic Types (e.g., T)
    Generic(String),

    // Type Alias (e.g., type AliasName = Type;)
    Alias(String, Box<AstType>),

    Function(Vec<AstType>, Box<AstType>), // (parameter_types, return_type)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

// Sample utility func to print AST
impl Program {
    pub fn new(statements: Vec<Stmt>) -> Self {
        Program { statements }
    }
}
