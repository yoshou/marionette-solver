use na::Scalar;
use std::ops::AddAssign;

extern crate nalgebra as na;
use nalgebra_sparse::coo::CooMatrix;

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

        for row_data in &mut scaled.rows {
            for column_data in &mut row_data.columns {
                let j = column_data.column;
                let block = &column_data.data;

                let scale_vec = scale.rows(j, block.ncols());
                column_data.data = block * na::DMatrix::from_diagonal(&scale_vec);
            }
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
        }

        column_norm_squared
    }

    pub fn gramian(&self) -> CsrBlockMatrix<T> {
        use std::collections::HashMap;

        let mut producted_terms = HashMap::<(usize, usize), na::DMatrix<T>>::new();

        for row_data in &self.rows {
            for column_data1 in &row_data.columns {
                let j1 = column_data1.column;
                let block1 = &column_data1.data;

                for column_data2 in &row_data.columns {
                    let j2 = column_data2.column;
                    let block2 = &column_data2.data;

                    let prod = block1.transpose() * block2;

                    if let Some(m) = producted_terms.get_mut(&(j1, j2)) {
                        m.add_assign(prod);
                    } else {
                        producted_terms.insert((j1, j2), prod);
                    }
                }
            }
        }

        let mut vec: Vec<_> = producted_terms.into_iter().collect();
        vec.sort_by(|(a, _), (b, _)| (a.0 * self.num_cols + a.1).cmp(&(b.0 * self.num_cols + b.1)));

        let mut gram = CsrBlockMatrix::new();

        let mut row = 0;
        let mut columns = Vec::<CsrColumnData<na::DMatrix<T>>>::new();
        for ((i, j), m) in vec {
            if i != row {
                let num_row = columns.first().unwrap().data.nrows();

                assert!(i == row + num_row);

                gram.add_row(num_row);

                for column in columns {
                    gram.add_row_block(column.column, &column.data);
                }

                columns = Vec::<CsrColumnData<na::DMatrix<T>>>::new();
            }
            let col_data = CsrColumnData::<na::DMatrix<T>> { column: j, data: m };

            columns.push(col_data);
            row = i;
        }

        if !columns.is_empty() {
            let num_row = columns.first().unwrap().data.nrows();
            gram.add_row(num_row);

            for column in columns {
                gram.add_row_block(column.column, &column.data);
            }
        }

        gram
    }

    pub fn to_dense_matrix(&self) -> na::DMatrix<T> {
        let mut m = na::DMatrix::<T>::zeros(self.num_rows, self.num_cols);

        let mut i = 0;
        for row_data in &self.rows {
            for column_data in &row_data.columns {
                let j = column_data.column;

                m.view_range_mut(
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
