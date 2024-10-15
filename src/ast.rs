// src/ast.rs

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum AstType {
    Int,
    Float,
    Bool,
    Char,
    Custom(String),             // For typedefs and user-defined structs
    Array(Box<AstType>, usize), // Element type and size
}

#[derive(Debug, Clone)]
pub struct Typedef {
    pub new_type: String,
    pub existing_type: AstType,
}

#[derive(Debug, Clone)]
pub struct AstStructField {
    pub name: String,
    pub ty: AstType,
}

#[derive(Debug, Clone)]
pub struct AstStruct {
    pub name: String,
    pub fields: Vec<AstStructField>,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub ty: AstType,
}

#[derive(Debug, Clone)]
pub struct AstFunction {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: AstType,
    pub body: Vec<AstStmt>,
}

#[derive(Debug, Clone)]
pub enum AstStmt {
    VariableDeclaration {
        name: String,
        ty: AstType,
        value: Option<AstExpr>,
    },
    Assignment {
        name: String,
        value: AstExpr,
    },
    Return(AstExpr),
    If {
        condition: AstExpr,
        then_branch: Vec<AstStmt>,
        else_branch: Option<Vec<AstStmt>>,
    },
    While {
        condition: AstExpr,
        body: Vec<AstStmt>,
    },
    Expression(AstExpr),
}

#[derive(Debug, Clone)]
pub enum AstExpr {
    Literal(Literal),
    Variable(String),
    BinaryOperation {
        left: Box<AstExpr>,
        operator: AstOperator,
        right: Box<AstExpr>,
    },
    UnaryOperation {
        operator: UnaryOperator,
        operand: Box<AstExpr>,
    },
    FunctionCall {
        name: String,
        args: Vec<AstExpr>,
    },
    // Additional expressions can be added here
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Char(char),
    String(String),
}

#[derive(Debug, Clone, Copy)]
pub enum AstOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOperator {
    Negate,
    Not,
    BitwiseNot,
}

// Helper type for variable mapping
pub type VariableMap = HashMap<String, cranelift::prelude::Variable>;
