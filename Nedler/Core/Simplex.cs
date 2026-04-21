using System;
using System.Linq;

namespace NelderMeadMethod.Core
{
    public class Simplex
    {
        public double[][] Vertexes { get; private set; }
        public double[] Values { get; private set; }
        public int Dimension { get; }
        public int VertexCount { get; }

        public Simplex(double[] startPoint, double step)
        {
            if (startPoint == null)
                throw new ArgumentNullException(nameof(startPoint));

            if (startPoint.Length == 0)
                throw new ArgumentException("Начальная точка не должна быть пустой.", nameof(startPoint));

            if (step <= 0)
                throw new ArgumentOutOfRangeException(nameof(step));

            Dimension = startPoint.Length;
            VertexCount = Dimension + 1;

            Vertexes = new double[VertexCount][];
            Values = new double[VertexCount];

            Vertexes[0] = (double[])startPoint.Clone();

            for (int i = 1; i < VertexCount; i++)
            {
                Vertexes[i] = (double[])startPoint.Clone();
                Vertexes[i][i - 1] += step;
            }
        }

        public void UpdateValues(Func<double[], double> func)
        {
            for (int i = 0; i < VertexCount; i++)
                Values[i] = func(Vertexes[i]);
        }

        public void SortVertexes()
        {
            var sorted = Values
                .Select((v, i) => new { v, i })
                .OrderBy(x => x.v)
                .ToArray();

            Vertexes = sorted.Select(x => Vertexes[x.i]).ToArray();
            Values = sorted.Select(x => x.v).ToArray();
        }

        public double[] GetBestVertex() => (double[])Vertexes[0].Clone();
        public double[] GetWorstVertex() => (double[])Vertexes[^1].Clone();

        public double GetValue(int i) => Values[i];
        public double GetSecondWorstValue() => Values[^2];

        public void ReplaceWorstVertex(double[] vertex, double value)
        {
            Vertexes[^1] = (double[])vertex.Clone();
            Values[^1] = value;
        }

        public double[] CalculateCentroid()
        {
            double[] centroid = new double[Dimension];

            for (int i = 0; i < VertexCount - 1; i++)
                for (int j = 0; j < Dimension; j++)
                    centroid[j] += Vertexes[i][j];

            for (int j = 0; j < Dimension; j++)
                centroid[j] /= (VertexCount - 1);

            return centroid;
        }

        public bool IsConverged(double eps)
        {
            return Math.Abs(Values[^1] - Values[0]) < eps;
        }

        public void Shrink(Func<double[], double> func, double sigma)
        {
            var best = Vertexes[0];

            for (int i = 1; i < VertexCount; i++)
            {
                for (int j = 0; j < Dimension; j++)
                    Vertexes[i][j] = best[j] + sigma * (Vertexes[i][j] - best[j]);

                Values[i] = func(Vertexes[i]);
            }

            Values[0] = func(Vertexes[0]);
            SortVertexes();
        }
    }
}