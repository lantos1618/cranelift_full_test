// src/main.rs

mod ast_validator;
mod ast;
mod cl_codegen;
mod lexer;
mod token;
mod parser;
mod error;

use ast_validator::AstValidator;
use cl_codegen::CodeGenerator;
use lexer::Lexer;
use parser::Parser;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use error::{CompilerError, ErrorType};

fn main() {
    // Sample source code in your language
    let source_code = "
    ";

    // Step 1: Lexing
    let mut lexer = Lexer::new(source_code);
    let tokens = match lexer.tokenize() {
        Ok(tokens) => {
            println!("Tokens: {:?}", tokens);
            tokens
        }
        Err(e) => {
            eprintln!("{}", e.display());
            return;
        }
    };

    // Step 2: Parsing
    let mut parser = Parser::new(tokens, source_code.to_string());
    let program = match parser.parse() {
        Ok(program) => {
            println!("AST: {:?}", program);
            program
        }
        Err(e) => {
            eprintln!("{}", e.display());
            return;
        }
    };

    // Step 3: Validation
    let mut validator = AstValidator::new();
    if let Err(error) = validator.validate_program(&program) {
        eprintln!("Validation error: {:?}", error);
        return;
    }
    println!("AST is valid.");

    // Step 4: Code Generation
    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .expect("Failed to create JITBuilder");
    let mut jit_module = JITModule::new(jit_builder);
    let mut codegen = CodeGenerator::new(&mut jit_module);
    if let Err(error) = codegen.compile_program(&program) {
        eprintln!("Code generation error: {:?}", error);
        return;
    }
    println!("Code generation successful.");
}
