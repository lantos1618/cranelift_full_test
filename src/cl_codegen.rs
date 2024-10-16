// src/cl_codegen.rs

use cranelift::prelude::*;
use cranelift_module::{FuncId, FuncOrDataId, Linkage, Module};

use crate::ast::*;


pub struct CodeGenerator<'a, M: Module> {
    pub module: &'a mut M,
}

impl<'a, M: Module> CodeGenerator<'a, M> {
    pub fn new(module: &'a mut M) -> Self {
        Self { module }
    }

}
