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
        // Define a Point struct
        Point {
            x: int,
            y: int
        };

        // Function to calculate distance from origin
        fn distance(p: Point) -> int {
            let x_squared: int = p.x * p.x;
            let y_squared: int = p.y * p.y;
            return x_squared + y_squared;
        }

        // Main function
        fn main() -> int {
            // Create a point
            let point: Point = Point { 
                x: 3,
                y: 4 
            };

            // Calculate and return distance
            return distance(point);
        }
    "#;

    // Step 1: Lexing
    let mut lexer = Lexer::new(source_code);
    let tokens = match lexer.tokenize() {
        Ok(tokens) => {
            println!("Lexical analysis successful!");
            println!("Tokens: {:?}", tokens);
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
    match validator.validate_program(&program) {
        Ok(_) => println!("Semantic analysis successful!"),
        Err(e) => {
            eprintln!("Validation error: {}", e);
            return;
        }
    }

    // Step 4: Code Generation
    let isa = match cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(settings::builder())) {
            Ok(isa) => isa,
            Err(e) => {
                eprintln!("Failed to create ISA: {}", e);
                return;
            }
    };

    let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut jit_module = JITModule::new(jit_builder);
    let mut codegen = CodeGenerator::new(&mut jit_module);
    
    match codegen.compile_program(&program) {
        Ok(_) => {
            println!("Code generation successful!");
            println!("Compilation completed successfully!");
        },
        Err(e) => {
            eprintln!("Code generation error: {}", e);
            return;
        }
    }
}
