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
use cranelift_jit::JITBuilder;
use cranelift_jit::JITModule;
use cranelift_codegen::settings;

fn main() {
    // Sample source code demonstrating key language features
    let source_code = r#"
        // Struct definition
        struct Point {
            x: int,
            y: int
        }

        // Main function
        fn main() -> int {
            let p = Point { x: 1, y: 2 };
            p.x = 3;
            p.y = 4;
            return p.x + p.y;
        }
    "#;

    // Step 1: Lexing
    let mut lexer = Lexer::new(source_code);
    let tokens = match lexer.tokenize() {
        Ok(tokens) => {
            println!("Lexical analysis successful!");
            // println!("Tokens: {:?}", tokens);
            tokens
        }
        Err(e) => {
            eprintln!("Lexical error: {}", e);
            return;
        }
    };

    // Step 2: Parsing
    let mut parser = Parser::new(tokens, source_code.to_string());
    let program = match parser.parse() {
        Ok(program) => {
            println!("Parsing successful!");
            println!("AST: {:?}", program);
            program
        }
        Err(e) => {
            eprintln!("Parsing error: {}", e);
            return;
        }
    };

    // Step 3: Semantic Analysis
    let mut validator = AstValidator::new();
    if let Err(error) = validator.validate_program(&program) {
        eprintln!("Validation error: {}", error);
        return;
    }
    println!("Semantic analysis successful!");

    // Step 4: Code Generation
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(settings::builder()))
        .unwrap();

    let jit_builder =  JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    
    let mut jit_module = JITModule::new(jit_builder);
    let mut codegen = CodeGenerator::new(&mut jit_module);
    
    match codegen.compile_program(&program) {
        Ok(_) => println!("Code generation successful!"),
        Err(e) => {
            eprintln!("Code generation error: {}", e);
            return;
        }
    }

    println!("Compilation completed successfully!");
}
