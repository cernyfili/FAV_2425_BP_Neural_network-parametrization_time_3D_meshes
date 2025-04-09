using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<float>;

namespace compact_rep_lib.FeatureExtractor
{
    internal class PCA
    {
        public static Vector[] GetCenterFeatures(Vector3[][] pc, int dim)
        {
            Vector[] result = new Vector[pc[0].Length];

            Matrix m = Matrix.Build.Dense(pc.Length * 3, pc[0].Length);
            for (int i = 0; i < m.RowCount / 3; i++)
            {
                for (int j = 0; j < m.ColumnCount; j++)
                {
                    m[3 * i, j]     = pc[i][j].X;
                    m[3 * i + 1, j] = pc[i][j].Y;
                    m[3 * i + 2, j] = pc[i][j].Z;
                }
            }

            // compute mean
            Matrix mean = Matrix.Build.Dense(pc.Length * 3, 1);
            for (int i = 0; i < m.RowCount; i++)
            {
                for (int j = 0; j < m.ColumnCount; j++)
                {
                    mean[i, 0] += m[i, j];
                }
                mean[i, 0] /= m.ColumnCount;
            }

            // subtract mean
            for (int i = 0; i < m.ColumnCount; i++)
            {
                for (int j = 0; j < m.RowCount; j++)
                {
                    m[j, i] -= mean[j, 0];
                }  
            }

            // autocorrelation
            var mt = m.Transpose();
            var ac = m * mt;

            var evd = ac.Evd();
            var coefs = mt * evd.EigenVectors;

            for (int i = 0; i < result.Length; i++)
            {
                float[] vec = new float[dim];

                for (int j = 0; j < dim; j++)
                {
                    vec[j] = coefs[i, coefs.ColumnCount - (j + 1)];
                }

                result[i] = Vector.Build.DenseOfArray(vec);
            }

            return result;
        }
    }
}
