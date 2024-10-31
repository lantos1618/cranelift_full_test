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
        // Define the Point struct first
        struct Point {
            x: int,
            y: int,
        };

        // Now we can use Point in our functions
        fn distance(p: Point) -> int {
            let x_squared: int = p.x * p.x;
            let y_squared: int = p.y * p.y;
            return x_squared + y_squared;
        }

        fn main() -> int {
            let point: Point = Point { x: 3, y: 4 };
            return distance(point);
        }
    "#;

    // Create lexer and generate tokens
    let mut lexer = Lexer::new(source_code);
    let tokens = match lexer.tokenize() {
        Ok(tokens) => {
            println!("Lexical analysis successful!");
            // println!("Tokens: {:?}", tokens);
            tokens
        }
        Err(e) => {
            println!("Lexical error: {}", e);
            return;
        }
    };

    // Create parser and generate AST
    let mut parser = Parser::new(tokens, source_code.to_string());
    let program = match parser.parse() {
        Ok(ast) => {
            println!("Parsing successful!");
            println!("AST: {:?}", ast);
            ast
        }
        Err(e) => {
            println!("Parsing error: {}", e);
            return;
        }
    };

    // Step 3: Semantic Analysis
    let mut validator = AstValidator::new();
    match validator.validate_program(&program) {
        Ok(_) => println!("Semantic analysis successful!"),
        Err(e) => {
            println!("Semantic error: {}", e);
            return;
        }
    }

    // Generate code
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(settings::builder()))
        .unwrap();
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);
    let mut codegen = CodeGenerator::new(&mut module);

    match codegen.compile_program(&program) {
        Ok(_) => {
            if let Some(func_id) = codegen.get_function_id("main") {
                module.finalize_definitions().unwrap();
                let code = module.get_finalized_function(func_id);
                let main_fn: fn() -> i64 = unsafe { std::mem::transmute(code) };
                let result = main_fn();
                println!("Result: {}", result);
            }
        }
        Err(e) => println!("Code generation error: {}", e),
    }
}
