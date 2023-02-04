use na::Scalar;

extern crate nalgebra as na;
use nalgebra_sparse::{coo::CooMatrix};
use nalgebra_sparse::csc::CscMatrix;

#[derive(Clone)]
pub struct CsrColumnData<T> {
    pub column: usize,
    pub data: T,
}

#[derive(Clone)]
pub struct CsrRowData<T> {
    pub columns: Vec<CsrColumnData<T>>,
    pub num_block_rows: usize,
}
impl<T> CsrRowData<T> {
    pub fn new(num_block_rows: usize) -> Self {
        CsrRowData::<T> {
            columns: Vec::<CsrColumnData<T>>::new(),
            num_block_rows: num_block_rows,
        }
    }
}

#[derive(Clone)]
pub struct CsrBlockMatrix<T: Scalar> {
    pub rows: Vec<CsrRowData<na::DMatrix<T>>>,
    num_rows: usize,
    num_cols: usize,
}

impl<T: na::RealField + Copy> CsrBlockMatrix<T> {
    pub fn new() -> Self {
        CsrBlockMatrix::<T> {
            rows: Vec::<CsrRowData<na::DMatrix<T>>>::new(),
            num_rows: 0,
            num_cols: 0,
        }
    }

    pub fn add_row(&mut self, num_block_rows: usize) {
        self.rows
            .push(CsrRowData::<na::DMatrix<T>>::new(num_block_rows));
        self.num_rows = self.num_rows + num_block_rows;
    }

    pub fn add_row_block(&mut self, col: usize, block: &na::DMatrix<T>) -> bool {
        if let Some(row_data) = self.rows.last_mut() {
            if let Some(last) = row_data.columns.last() {
                if col >= last.column + last.data.ncols() {
                    let col_data = CsrColumnData::<na::DMatrix<T>> {
                        column: col,
                        data: block.clone(),
                    };
                    row_data.columns.push(col_data);

                    self.num_cols = self.num_cols.max(col + block.ncols());
                    return true;
                }
            } else {
                let col_data = CsrColumnData::<na::DMatrix<T>> {
                    column: col,
                    data: block.clone(),
                };
                row_data.columns.push(col_data);

                self.num_cols = self.num_cols.max(col + block.ncols());
                return true;
            }
        }

        false
    }

    pub fn scale_columns(&self, scale: &na::DVector<T>) -> CsrBlockMatrix<T> {
        let mut scaled = self.clone();

        let mut i = 0;
        for row_data in &mut scaled.rows {
            for column_data in &mut row_data.columns {
                let j = column_data.column;
                let block = &column_data.data;

                let scale_vec = scale.rows(j, block.ncols());
                column_data.data = block * na::DMatrix::from_diagonal(&scale_vec);
            }
            i += row_data.num_block_rows;
        }

        scaled
    }

    pub fn mul(&self, v: &na::DVector<T>) -> na::DVector<T> {
        let mut result = na::DVector::<T>::zeros(self.num_rows);

        let mut i = 0;
        for row_data in &self.rows {
            for column_data in &row_data.columns {
                let j = column_data.column;
                let block = &column_data.data;

                let block_vec = v.rows(j, block.ncols());

                let mut result_block_vec = result.rows_mut(i, block.nrows());
                result_block_vec.copy_from(&(&result_block_vec + block * block_vec));
            }
            i += row_data.num_block_rows;
        }

        result
    }

    pub fn transpose_and_mul(&self, v: &na::DVector<T>) -> na::DVector<T> {
        let mut result = na::DVector::<T>::zeros(self.num_cols);

        let mut i = 0;
        for row_data in &self.rows {
            for column_data in &row_data.columns {
                let j = column_data.column;
                let block = &column_data.data;

                let block_vec = v.rows(i, block.nrows());

                let mut result_block_vec = result.rows_mut(j, block.ncols());
                result_block_vec.copy_from(&(&result_block_vec + block.transpose() * block_vec));
            }
            i += row_data.num_block_rows;
        }

        result
    }

    pub fn column_norm_squared(&self) -> na::DVector<T> {
        let mut column_norm_squared = na::DVector::<T>::zeros(self.num_cols);

        let mut i = 0;
        for row_data in &self.rows {
            for column_data in &row_data.columns {
                let j = column_data.column;
                let block = &column_data.data;

                let block_column_norm_squared = na::DVector::from_iterator(
                    block.ncols(),
                    block.column_iter().map(|x| x.norm_squared()),
                );

                let mut block_vec = column_norm_squared.rows_mut(j, block.ncols());
                block_vec.copy_from(&(&block_vec + block_column_norm_squared));
            }
            i += row_data.num_block_rows;
        }

        column_norm_squared
    }

    pub fn to_dense_matrix(&self) -> na::DMatrix<T> {
        let mut m = na::DMatrix::<T>::zeros(self.num_rows, self.num_cols);

        let mut i = 0;
        for row_data in &self.rows {
            for column_data in &row_data.columns {
                let j = column_data.column;

                m.slice_range_mut(
                    i..(i + column_data.data.nrows()),
                    j..(j + column_data.data.ncols()),
                )
                .copy_from(&column_data.data);
            }
            i += row_data.num_block_rows;
        }

        m
    }

    pub fn to_sparse_matrix(&self) -> CooMatrix<T> {
        let mut coo = CooMatrix::<T>::new(self.num_rows, self.num_cols);

        let mut i = 0;
        for row_data in &self.rows {
            for column_data in &row_data.columns {
                let j = column_data.column;
                let block = &column_data.data;

                for ii in 0..column_data.data.nrows() {
                    for jj in 0..column_data.data.ncols() {
                        coo.push(ii + i, jj + j, block[(ii, jj)]);
                    }
                }
            }
            i += row_data.num_block_rows;
        }

        coo
    }

    pub fn ncols(&self) -> usize {
        self.num_cols
    }
    pub fn nrows(&self) -> usize {
        self.num_rows
    }
}
