// src/main.rs

mod ast_validator;
mod ast;
mod cl_codegen;
mod lexer;
mod token;
mod parser;

use ast_validator::AstValidator;
use cl_codegen::CodeGenerator;
use lexer::Lexer;
use parser::Parser;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;

fn main() {
    // Sample source code in your language
    let source_code = "
        fn add(a: i32, b: i32) -> i32 {
            return a + b;
        }

        fn main() -> i32 {
            let x: i32 = 5;
            let y: i32 = 3;
            let result: i32 = add(x, y * 2);
            return result;
        }
    ";

    // Step 1: Lexing
    let mut lexer = Lexer::new(source_code);
    let tokens = lexer.tokenize();
    println!("Tokens: {:?}", tokens);

    // Step 2: Parsing
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    println!("AST: {:?}", program);

    // Step 3: Validation
    let mut validator = AstValidator::new();
    if let Err(error) = validator.validate_program(&program) {
        println!("Validation error: {:?}", error);
        return;
    }
    println!("AST is valid.");

    // Step 4: Code Generation
    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .expect("Failed to create JITBuilder");
    let mut jit_module = JITModule::new(jit_builder);
    let mut codegen = CodeGenerator::new(&mut jit_module);
    if let Err(error) = codegen.compile_program(&program) {
        println!("Code generation error: {:?}", error);
        return;
    }
    println!("Code generation successful.");


}
