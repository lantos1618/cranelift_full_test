// src/main.rs

mod ast_validator;
mod ast;
mod cl_codegen;
mod lexer;
mod token;
mod parser;

use cranelift::codegen::settings;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use target_lexicon::Triple;

use ast::*;

use cl_codegen::CodeGenerator;
use ast_validator::AstValidator;
use lexer::Lexer;
use parser::Parser;



fn main() {
    println!("Hello, world!");
}
