using compact_rep_lib.Structures;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.Serialization.Formatters;
using System.Text;
using System.Threading.Tasks;

using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<float>;

namespace compact_rep_lib.FeatureExtractor
{
    public class PCAFeatureExtractor : IFeatureExtractor
    {
        private readonly Vector3[][] pc;
        private readonly Vector[] pcaFeatures;
        private readonly int dim;
        private readonly int k;
        private readonly int[][] neighbors;

        const float SHAPE = 2f;
        const float LIM_EPS = 1e-6f;
        //private Rgb24[] pcCol;

        public PCAFeatureExtractor(Vector3[][] pc, int dim, int k)
        {
            this.pc = pc;
            this.pcaFeatures = PCA.GetCenterFeatures(pc, dim);
            this.dim = dim;
            this.k = k;
            //pcCol = GetCenterColors();
            this.neighbors = EstablishCenterNeighborhood(pc);
        }

        public int[][] EstablishCenterNeighborhood(Vector3[][] pc)
        {
            int n = pc[0].Length;
            int[][] result = new int[pc[0].Length][];
            float[][] dists = new float[n][];

            for (int i = 0; i < n; i++)
            {
                dists[i] = new float[n];
            }

            Parallel.For(0, n, (int i) => {
                for (int j = i + 1; j < n; j++)
                {
                    float maxDist = 0f;

                    for (int frame = 0; frame < pc.Length; frame++)
                    {
                        float dist = Vector3.DistanceSquared(pc[frame][i], pc[frame][j]);

                        if (dist > maxDist)
                        {
                            maxDist = dist;
                        }
                    }

                    dists[i][j] = maxDist;
                    dists[j][i] = maxDist;
                }
            });

            for (int i = 0; i < n; i++)
            {
                result[i] = dists[i].Select((affinity, index) => (affinity, index)).OrderBy(x => x.affinity).Take(k).Select(x => x.index).ToArray();
            }

            return result;
        }

        //private Rgb24[] GetCenterColors()
        //{
        //    if (dim < 3)
        //    {
        //        throw new Exception("Not enough of dimensions to calculate colors");
        //    }

        //    int n = pcaFeatures.Length;

        //    Rgb24[] colors = new Rgb24[n];
        //    Vector3[] threeDimFeatures = new Vector3[n];

        //    Vector3 min;
        //    Vector3 max;

        //    threeDimFeatures[0] = new Vector3(pcaFeatures[0][0], pcaFeatures[0][1], pcaFeatures[0][2]);
        //    min = threeDimFeatures[0];
        //    max = threeDimFeatures[0];

        //    for (int i = 1; i < n; i++)
        //    {
        //        threeDimFeatures[i] = new Vector3(pcaFeatures[i][0], pcaFeatures[i][1], pcaFeatures[i][2]);
        //        min = Vector3.Min(min, threeDimFeatures[i]);
        //        max = Vector3.Max(max, threeDimFeatures[i]);
        //    }

        //    for (int i = 0; i < n; i++)
        //    {
        //        colors[i] = Vector2Color(Map(threeDimFeatures[i], min, max));
        //    }

        //    return colors;
        //}

        //private static Vector3 Map(Vector3 p, Vector3 min, Vector3 max)
        //{
        //    return (p - min) / (max - min);
        //}

        //private static Rgb24 Vector2Color(Vector3 vec)
        //{
        //    byte r = (byte)(vec.X * 255);
        //    byte g = (byte)(vec.Y * 255);
        //    byte b = (byte)(vec.Z * 255);

        //    return new Rgb24(r, g, b);
        //}

        //public void ColorizeVoronoi(List<Sample> samples, int frame)
        //{
        //    Vector3[] framePC = pc[frame];

        //    KDTree tree = new KDTree(framePC);

        //    Parallel.For(0, samples.Count, (i) => {
        //        Sample s = samples[i];

        //        int nn = tree.findNearest(s.position, out _);

        //        s.colour = pcCol[nn];
        //    });
        //}

        public void ExtractFeaturesHachaWeights(List<Sample> samples, int frame)
        {
            Vector3[] framePC = pc[frame];

            KDTree tree = new(framePC);

            Parallel.For(0, samples.Count, (i) => {
                Sample s = samples[i];

                int nearest = tree.findNearest(s.position, out float dist);
                float[] distances = new float[k];
                int[] centers = neighbors[nearest];


                for (int centerIndex = 0; centerIndex < k; centerIndex++)
                {
                    distances[centerIndex] = Vector3.Distance(s.position, framePC[centers[centerIndex]]);
                }

                float[] softMin = new float[k];
                float softMinSum = 0f;
                float softMinMin = float.MaxValue;

                for (int j = 0; j < k; j++)
                {
                    softMin[j] = MathF.Exp(-distances[j] / (SHAPE * dist + LIM_EPS));
                    softMinSum += softMin[j];
                }

                for (int j = 0; j < k; j++)
                {
                    softMin[j] /= softMinSum;

                    if (softMin[j] < softMinMin)
                    {
                        softMinMin = softMin[j];
                    }
                }

                float[] weights = new float[k];
                float weightSum = 0f;

                for (int j = 0; j < k; ++j)
                {
                    weights[j] = softMin[j] - softMinMin + 1e-6f;
                    weightSum += weights[j];
                }

                Vector feature = Vector.Build.Dense(dim);
                float invWeightSum = 1 / weightSum;

                for (int j = 0; j < k; ++j)
                {

                    feature += weights[j] * invWeightSum * pcaFeatures[centers[j]];
                }

                s.feature = [..feature];
            });
        }

        public void ExtractFeaturesEDWeights(List<Sample> samples, int frame)
        {
            Vector3[] framePC = pc[frame];

            KDTree tree = new(framePC);

            Parallel.For(0, samples.Count, (i) => {
                Sample s = samples[i];

                List<(int, float)> knn = tree.kNN(s.position, k);

                Vector feature = Vector.Build.Dense(dim);

                float maxDist = knn[k - 1].Item2;

                for (int j = 0; j < k - 1; j++)
                {
                    var neighbour = knn[j];
                    float weight = 1f - neighbour.Item2 / maxDist;

                    feature += (weight * weight) * pcaFeatures[neighbour.Item1];
                }

                s.feature = [..feature];
            });
        }

        public void ExtractFeatures(List<Sample> samples, int frame)
        {
            //ExtractFeaturesHachaWeights(samples, frame);
            ExtractFeaturesEDWeights(samples, frame);
        }
    }
}
