use na::Scalar;
use num_traits::{Zero};



extern crate nalgebra as na;

struct CsrColumnData<T> {
    column: usize,
    data: T,
}

struct CsrRowData<T> {
    columns: Vec<CsrColumnData<T>>,
    num_block_rows: usize,
}
impl<T> CsrRowData<T> {
    pub fn new(num_block_rows: usize) -> Self {
        CsrRowData::<T> {
            columns: Vec::<CsrColumnData<T>>::new(),
            num_block_rows: num_block_rows,
        }
    }
}

pub struct CsrBlockMatrix<T : Scalar> {
    rows: Vec<CsrRowData<na::DMatrix<T>>>,
    num_rows: usize,
    num_cols: usize,
}

impl<T : Scalar + Zero> CsrBlockMatrix<T> {
    pub fn new() -> Self {
        CsrBlockMatrix::<T> {
            rows: Vec::<CsrRowData<na::DMatrix<T>>>::new(),
            num_rows: 0,
            num_cols: 0,
        }
    }

    pub fn add_row(&mut self, num_block_rows: usize) {
        self.rows.push(CsrRowData::<na::DMatrix<T>>::new(num_block_rows));
        self.num_rows = self.num_rows + num_block_rows;
    }

    pub fn add_row_block(&mut self, col: usize, block: &na::DMatrix<T>) -> bool {
        if let Some(row_data) = self.rows.last_mut() {
            if let Some(last) = row_data.columns.last() {
                if col >= last.column + last.data.ncols() {
                    let col_data = CsrColumnData::<na::DMatrix<T>> {column: col, data: block.clone()};
                    row_data.columns.push(col_data);

                    self.num_cols = self.num_cols.max(col + block.ncols());
                    return true;
                }
            } else {
                let col_data = CsrColumnData::<na::DMatrix<T>> {column: col, data: block.clone()};
                row_data.columns.push(col_data);

                self.num_cols = self.num_cols.max(col + block.ncols());
                return true;
            }
        }

        false
    }

    pub fn to_dense_matrix(&self) -> na::DMatrix<T> {
        let mut m = na::DMatrix::<T>::zeros(self.num_rows, self.num_cols);

        let mut i = 0;
        for row_data in &self.rows {
            for column_data in &row_data.columns {
                let j = column_data.column;

                m.slice_range_mut(i..(i + column_data.data.nrows()), j..(j + column_data.data.ncols())).copy_from(&column_data.data);
            }
            i += row_data.num_block_rows;
        }

        m
    }
}
