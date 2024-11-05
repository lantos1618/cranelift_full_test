// src/main.rs


use cranelift_jit::JITBuilder;
use cranelift_jit::JITModule;
use cranelift_codegen::settings;

use cranelift_test::lexer::Lexer;
use cranelift_test::parser::Parser;
use cranelift_test::ast_validator::AstValidator;
use cranelift_test::cl_codegen::CodeGenerator;

fn main() {
    // Sample source code demonstrating key language features
    let source_code = r#"
        fn fib(n: int) -> int {
            if (n <= 1) {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }

        struct Point {
            x: int,
            y: int,
        };

        fn distance(p: Point) -> int {
            let x_squared: int = p.x * p.x;
            let y_squared: int = p.y * p.y;
            return x_squared + y_squared;
        }

        fn main() -> int {
            let point: Point = Point { x: 3, y: 4 };
            return distance(point) + fib(10);
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
