// src/main.rs

mod ast;
mod cl_codegen;
mod intergrated_tests;

use cranelift::codegen::settings;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use target_lexicon::Triple;

use ast::*;
use cl_codegen::CodeGenerator;

fn main() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    // Create an instance of the code generator
    let mut codegen = CodeGenerator::new(&mut module);

    // Define the `add` function
    let add_function = AstFunction {
        name: "add".to_string(),
        params: vec![
            Parameter {
                name: "a".to_string(),
                ty: AstType::Int,
            },
            Parameter {
                name: "b".to_string(),
                ty: AstType::Int,
            },
        ],
        return_type: AstType::Int,
        body: vec![AstStmt::Return(AstExpr::BinaryOperation {
            left: Box::new(AstExpr::Variable("a".to_string())),
            operator: AstOperator::Add,
            right: Box::new(AstExpr::Variable("b".to_string())),
        })],
    };

    // Define the `main` function
    let main_function = AstFunction {
        name: "main".to_string(),
        params: vec![],
        return_type: AstType::Int,
        body: vec![
            AstStmt::VariableDeclaration {
                name: "result".to_string(),
                ty: AstType::Int,
                value: Some(AstExpr::FunctionCall {
                    name: "add".to_string(),
                    args: vec![
                        AstExpr::Literal(Literal::Int(5)),
                        AstExpr::Literal(Literal::Int(3)),
                    ],
                }),
            },
            AstStmt::Return(AstExpr::Variable("result".to_string())),
        ],
    };

    // Compile the functions
    codegen.compile_function(&add_function);
    let main_func_id = codegen.compile_function(&main_function);

    // Finalize the functions
    module.finalize_definitions();

    // Get a pointer to the `main` function
    let main_func = module.get_finalized_function(main_func_id);

    // Cast the pointer to a callable function
    let main_fn = unsafe { std::mem::transmute::<_, fn() -> i64>(main_func) };

    // Execute the `main` function
    let result = main_fn();

    println!("Result from main(): {}", result);
}
