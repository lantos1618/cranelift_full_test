// tests/integration_test.rs

use crate::ast::*;
use crate::cl_codegen::CodeGenerator;
use cranelift::codegen::settings;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use target_lexicon::Triple;

#[test]
fn test_add_function() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

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

    // Compile the `add` function
    let add_func_id = codegen.compile_function(&add_function);

    // Finalize the function
    module.finalize_definitions();

    // Get a pointer to the `add` function
    let add_func = module.get_finalized_function(add_func_id);

    // Cast the pointer to a callable function
    let add_fn = unsafe { std::mem::transmute::<_, fn(i64, i64) -> i64>(add_func) };

    // Test the `add` function
    let result = add_fn(5, 3);
    assert_eq!(result, 8);
}

#[test]
fn test_variable_assignment_and_return() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    let mut codegen = CodeGenerator::new(&mut module);

    // Define a function that assigns variables and returns the result
    let func = AstFunction {
        name: "variable_test".to_string(),
        params: vec![],
        return_type: AstType::Int,
        body: vec![
            AstStmt::VariableDeclaration {
                name: "x".to_string(),
                ty: AstType::Int,
                value: Some(AstExpr::Literal(Literal::Int(10))),
            },
            AstStmt::Assignment {
                name: "x".to_string(),
                value: AstExpr::BinaryOperation {
                    left: Box::new(AstExpr::Variable("x".to_string())),
                    operator: AstOperator::Multiply,
                    right: Box::new(AstExpr::Literal(Literal::Int(2))),
                },
            },
            AstStmt::Return(AstExpr::Variable("x".to_string())),
        ],
    };

    // Compile the function
    let func_id = codegen.compile_function(&func);

    // Finalize the function
    module.finalize_definitions();

    // Get a pointer to the function
    let func_ptr = module.get_finalized_function(func_id);

    // Cast the pointer to a callable function
    let func_fn = unsafe { std::mem::transmute::<_, fn() -> i64>(func_ptr) };

    // Test the function
    let result = func_fn();
    assert_eq!(result, 20);
}

#[test]
fn test_if_statement() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    let mut codegen = CodeGenerator::new(&mut module);

    // Define a function that uses an if statement
    let func = AstFunction {
        name: "if_statement_test".to_string(),
        params: vec![],
        return_type: AstType::Int,
        body: vec![
            AstStmt::VariableDeclaration {
                name: "x".to_string(),
                ty: AstType::Int,
                value: Some(AstExpr::Literal(Literal::Int(5))),
            },
            AstStmt::If {
                condition: AstExpr::BinaryOperation {
                    left: Box::new(AstExpr::Variable("x".to_string())),
                    operator: AstOperator::Greater,
                    right: Box::new(AstExpr::Literal(Literal::Int(3))),
                },
                then_branch: vec![AstStmt::Return(AstExpr::Literal(Literal::Int(1)))],
                else_branch: Some(vec![AstStmt::Return(AstExpr::Literal(Literal::Int(0)))]),
            },
        ],
    };

    // Compile the function
    let func_id = codegen.compile_function(&func);

    // Finalize the function
    module.finalize_definitions();

    // Get a pointer to the function
    let func_ptr = module.get_finalized_function(func_id);

    // Cast the pointer to a callable function
    let func_fn = unsafe { std::mem::transmute::<_, fn() -> i64>(func_ptr) };

    // Test the function
    let result = func_fn();
    assert_eq!(result, 1);
}

#[test]
fn test_while_loop() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    let mut codegen = CodeGenerator::new(&mut module);

    // Define a function that uses a while loop to compute factorial
    let func = AstFunction {
        name: "factorial".to_string(),
        params: vec![],
        return_type: AstType::Int,
        body: vec![
            AstStmt::VariableDeclaration {
                name: "n".to_string(),
                ty: AstType::Int,
                value: Some(AstExpr::Literal(Literal::Int(5))),
            },
            AstStmt::VariableDeclaration {
                name: "result".to_string(),
                ty: AstType::Int,
                value: Some(AstExpr::Literal(Literal::Int(1))),
            },
            AstStmt::While {
                condition: AstExpr::BinaryOperation {
                    left: Box::new(AstExpr::Variable("n".to_string())),
                    operator: AstOperator::Greater,
                    right: Box::new(AstExpr::Literal(Literal::Int(1))),
                },
                body: vec![
                    AstStmt::Assignment {
                        name: "result".to_string(),
                        value: AstExpr::BinaryOperation {
                            left: Box::new(AstExpr::Variable("result".to_string())),
                            operator: AstOperator::Multiply,
                            right: Box::new(AstExpr::Variable("n".to_string())),
                        },
                    },
                    AstStmt::Assignment {
                        name: "n".to_string(),
                        value: AstExpr::BinaryOperation {
                            left: Box::new(AstExpr::Variable("n".to_string())),
                            operator: AstOperator::Subtract,
                            right: Box::new(AstExpr::Literal(Literal::Int(1))),
                        },
                    },
                ],
            },
            AstStmt::Return(AstExpr::Variable("result".to_string())),
        ],
    };

    // Compile the function
    let func_id = codegen.compile_function(&func);

    // Finalize the function
    module.finalize_definitions();

    // Get a pointer to the function
    let func_ptr = module.get_finalized_function(func_id);

    // Cast the pointer to a callable function
    let func_fn = unsafe { std::mem::transmute::<_, fn() -> i64>(func_ptr) };

    // Test the function
    let result = func_fn();
    assert_eq!(result, 120); // 5! = 120
}

#[test]
fn test_recursive_function() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    let mut codegen = CodeGenerator::new(&mut module);

    // Define a recursive function to compute Fibonacci numbers
    let fib_function = AstFunction {
        name: "fib".to_string(),
        params: vec![
            Parameter {
                name: "n".to_string(),
                ty: AstType::Int,
            },
        ],
        return_type: AstType::Int,
        body: vec![
            AstStmt::If {
                condition: AstExpr::BinaryOperation {
                    left: Box::new(AstExpr::Variable("n".to_string())),
                    operator: AstOperator::LessEqual,
                    right: Box::new(AstExpr::Literal(Literal::Int(1))),
                },
                then_branch: vec![AstStmt::Return(AstExpr::Variable("n".to_string()))],
                else_branch: Some(vec![AstStmt::Return(AstExpr::BinaryOperation {
                    left: Box::new(AstExpr::FunctionCall {
                        name: "fib".to_string(),
                        args: vec![AstExpr::BinaryOperation {
                            left: Box::new(AstExpr::Variable("n".to_string())),
                            operator: AstOperator::Subtract,
                            right: Box::new(AstExpr::Literal(Literal::Int(1))),
                        }],
                    }),
                    operator: AstOperator::Add,
                    right: Box::new(AstExpr::FunctionCall {
                        name: "fib".to_string(),
                        args: vec![AstExpr::BinaryOperation {
                            left: Box::new(AstExpr::Variable("n".to_string())),
                            operator: AstOperator::Subtract,
                            right: Box::new(AstExpr::Literal(Literal::Int(2))),
                        }],
                    }),
                })]),
            },
        ],
    };

    // Compile the `fib` function
    let fib_func_id = codegen.compile_function(&fib_function);

    // Finalize the function
    module.finalize_definitions();

    // Get a pointer to the `fib` function
    let fib_func = module.get_finalized_function(fib_func_id);

    // Cast the pointer to a callable function
    let fib_fn = unsafe { std::mem::transmute::<_, fn(i64) -> i64>(fib_func) };

    // Test the `fib` function
    let result = fib_fn(10);
    assert_eq!(result, 55); // fib(10) = 55
}

#[test]
fn test_logical_operations() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    let mut codegen = CodeGenerator::new(&mut module);

    // Define a function that tests logical operations
    let func = AstFunction {
        name: "logical_operations_test".to_string(),
        params: vec![],
        return_type: AstType::Int,
        body: vec![
            AstStmt::VariableDeclaration {
                name: "a".to_string(),
                ty: AstType::Int,
                value: Some(AstExpr::Literal(Literal::Int(1))),
            },
            AstStmt::VariableDeclaration {
                name: "b".to_string(),
                ty: AstType::Int,
                value: Some(AstExpr::Literal(Literal::Int(0))),
            },
            AstStmt::If {
                condition: AstExpr::BinaryOperation {
                    left: Box::new(AstExpr::Variable("a".to_string())),
                    operator: AstOperator::And,
                    right: Box::new(AstExpr::Variable("b".to_string())),
                },
                then_branch: vec![AstStmt::Return(AstExpr::Literal(Literal::Int(1)))],
                else_branch: Some(vec![AstStmt::Return(AstExpr::Literal(Literal::Int(0)))]),
            },
        ],
    };

    // Compile the function
    let func_id = codegen.compile_function(&func);

    // Finalize the function
    module.finalize_definitions();

    // Get a pointer to the function
    let func_ptr = module.get_finalized_function(func_id);

    // Cast the pointer to a callable function
    let func_fn = unsafe { std::mem::transmute::<_, fn() -> i64>(func_ptr) };

    // Test the function
    let result = func_fn();
    assert_eq!(result, 0);
}

#[test]
fn test_function_call_with_arguments() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    let mut codegen = CodeGenerator::new(&mut module);

    // Define a function that increments its argument
    let increment_function = AstFunction {
        name: "increment".to_string(),
        params: vec![
            Parameter {
                name: "x".to_string(),
                ty: AstType::Int,
            },
        ],
        return_type: AstType::Int,
        body: vec![AstStmt::Return(AstExpr::BinaryOperation {
            left: Box::new(AstExpr::Variable("x".to_string())),
            operator: AstOperator::Add,
            right: Box::new(AstExpr::Literal(Literal::Int(1))),
        })],
    };

    // Define a function that calls `increment`
    let func = AstFunction {
        name: "test_increment".to_string(),
        params: vec![],
        return_type: AstType::Int,
        body: vec![AstStmt::Return(AstExpr::FunctionCall {
            name: "increment".to_string(),
            args: vec![AstExpr::Literal(Literal::Int(41))],
        })],
    };

    // Compile the functions
    codegen.compile_function(&increment_function);
    let func_id = codegen.compile_function(&func);

    // Finalize the functions
    module.finalize_definitions();

    // Get a pointer to the function
    let func_ptr = module.get_finalized_function(func_id);

    // Cast the pointer to a callable function
    let func_fn = unsafe { std::mem::transmute::<_, fn() -> i64>(func_ptr) };

    // Test the function
    let result = func_fn();
    assert_eq!(result, 42);
}

#[test]
fn test_nested_if_else() {
    // Set up Cranelift JIT module
    let triple = Triple::host();
    let isa_builder = cranelift::codegen::isa::lookup(triple).unwrap();
    let flag_builder = settings::builder();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    let mut codegen = CodeGenerator::new(&mut module);

    // Define a function with nested if-else statements
    let func = AstFunction {
        name: "nested_if_else_test".to_string(),
        params: vec![],
        return_type: AstType::Int,
        body: vec![
            AstStmt::VariableDeclaration {
                name: "x".to_string(),
                ty: AstType::Int,
                value: Some(AstExpr::Literal(Literal::Int(10))),
            },
            AstStmt::If {
                condition: AstExpr::BinaryOperation {
                    left: Box::new(AstExpr::Variable("x".to_string())),
                    operator: AstOperator::GreaterEqual,
                    right: Box::new(AstExpr::Literal(Literal::Int(10))),
                },
                then_branch: vec![AstStmt::If {
                    condition: AstExpr::BinaryOperation {
                        left: Box::new(AstExpr::Variable("x".to_string())),
                        operator: AstOperator::Equal,
                        right: Box::new(AstExpr::Literal(Literal::Int(10))),
                    },
                    then_branch: vec![AstStmt::Return(AstExpr::Literal(Literal::Int(1)))],
                    else_branch: Some(vec![AstStmt::Return(AstExpr::Literal(Literal::Int(2)))]),
                }],
                else_branch: Some(vec![AstStmt::Return(AstExpr::Literal(Literal::Int(0)))]),
            },
        ],
    };

    // Compile the function
    let func_id = codegen.compile_function(&func);

    // Finalize the function
    module.finalize_definitions();

    // Get a pointer to the function
    let func_ptr = module.get_finalized_function(func_id);

    // Cast the pointer to a callable function
    let func_fn = unsafe { std::mem::transmute::<_, fn() -> i64>(func_ptr) };

    // Test the function
    let result = func_fn();
    assert_eq!(result, 1);
}
