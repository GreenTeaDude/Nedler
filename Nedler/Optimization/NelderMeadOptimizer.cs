using System;
using NelderMeadMethod.Core;
using System.Collections.Generic;

namespace NelderMeadMethod.Optimization
{
    public class NelderMeadOptimizer
    {
        public double ReflectionCoeff { get; set; } = 1.0;
        public double ExpansionCoeff { get; set; } = 2.0;
        public double ContractionCoeff { get; set; } = 0.5;
        public double ShrinkCoeff { get; set; } = 0.5;
        public double Step { get; set; } = 0.5;
        public double Accuracy { get; set; } = 1e-6;

        public List<double> HistoryValues { get; } = new List<double>();
        public List<SimplexSnapshot> HistorySimplex { get; } = new List<SimplexSnapshot>();

        private readonly Func<double[], double> objectiveFunction;

        public NelderMeadOptimizer(Func<double[], double> objectiveFunction)
        {
            this.objectiveFunction = objectiveFunction ?? throw new ArgumentNullException(nameof(objectiveFunction));
        }

        public double[] Optimize(double[] startPoint, int maxIterations)
        {
            var simplex = new Simplex(startPoint, Step);
            simplex.UpdateValues(objectiveFunction);
            simplex.SortVertexes();

            
            SaveSimplexSnapshot(simplex, 0);

            for (int iter = 0; iter < maxIterations; iter++)
            {
                HistoryValues.Add(simplex.GetValue(0));

                if (simplex.IsConverged(Accuracy))
                    break;

                var centroid = simplex.CalculateCentroid();
                var worst = simplex.GetWorstVertex();

                double best = simplex.GetValue(0);
                double secondWorst = simplex.GetSecondWorstValue();
                double worstVal = simplex.GetValue(simplex.VertexCount - 1);

                double[] reflected = new double[simplex.Dimension];
                for (int i = 0; i < simplex.Dimension; i++)
                    reflected[i] = centroid[i] + ReflectionCoeff * (centroid[i] - worst[i]);

                double reflectedVal = objectiveFunction(reflected);

                if (reflectedVal < best)
                {
                    double[] expanded = new double[simplex.Dimension];
                    for (int i = 0; i < simplex.Dimension; i++)
                        expanded[i] = centroid[i] + ExpansionCoeff * (reflected[i] - centroid[i]);

                    double expandedVal = objectiveFunction(expanded);

                    if (expandedVal < reflectedVal)
                        simplex.ReplaceWorstVertex(expanded, expandedVal);
                    else
                        simplex.ReplaceWorstVertex(reflected, reflectedVal);

                    simplex.SortVertexes();
                    SaveSimplexSnapshot(simplex, iter + 1);
                    continue;
                }

                if (reflectedVal < secondWorst)
                {
                    simplex.ReplaceWorstVertex(reflected, reflectedVal);
                    simplex.SortVertexes();
                    SaveSimplexSnapshot(simplex, iter + 1);
                    continue;
                }

                double[] contracted = new double[simplex.Dimension];

                if (reflectedVal < worstVal)
                {
                    for (int i = 0; i < simplex.Dimension; i++)
                        contracted[i] = centroid[i] + ContractionCoeff * (reflected[i] - centroid[i]);
                }
                else
                {
                    for (int i = 0; i < simplex.Dimension; i++)
                        contracted[i] = centroid[i] - ContractionCoeff * (centroid[i] - worst[i]);
                }

                double contractedVal = objectiveFunction(contracted);

                if (contractedVal < worstVal)
                {
                    simplex.ReplaceWorstVertex(contracted, contractedVal);
                    simplex.SortVertexes();
                    SaveSimplexSnapshot(simplex, iter + 1);
                    continue;
                }

                simplex.Shrink(objectiveFunction, ShrinkCoeff);
                SaveSimplexSnapshot(simplex, iter + 1);
            }

            return simplex.GetBestVertex();
        }

        private void SaveSimplexSnapshot(Simplex simplex, int iteration)
        {
            var snapshot = new SimplexSnapshot
            {
                Iteration = iteration,
                VertexCount = simplex.VertexCount,
                Dimension = simplex.Dimension
            };

            snapshot.Vertexes = new double[simplex.VertexCount][];
            for (int i = 0; i < simplex.VertexCount; i++)
            {
                snapshot.Vertexes[i] = (double[])simplex.Vertexes[i].Clone();
            }

            snapshot.Values = new double[simplex.VertexCount];
            for (int i = 0; i < simplex.VertexCount; i++)
            {
                snapshot.Values[i] = simplex.Values[i];
            }

            HistorySimplex.Add(snapshot);
        }

        public void SaveSimplexHistoryToFile(string filename)
        {
            using (var writer = new System.IO.StreamWriter(filename))
            {
                
                writer.WriteLine("# Simplex evolution history");
                writer.WriteLine($"# Total iterations: {HistorySimplex.Count}");
                writer.WriteLine("# Format: iteration,vertex_index,x1,x2,...,xn,function_value");
                writer.WriteLine("# ---");

                foreach (var snapshot in HistorySimplex)
                {
                    for (int i = 0; i < snapshot.VertexCount; i++)
                    {
                        writer.Write($"{snapshot.Iteration},{i}");
                        for (int j = 0; j < snapshot.Dimension; j++)
                        {
                            writer.Write($",{snapshot.Vertexes[i][j].ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                        }
                        writer.WriteLine($",{snapshot.Values[i].ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                    }
                }
            }
        }
    }

    public class SimplexSnapshot
    {
        public int Iteration { get; set; }
        public int VertexCount { get; set; }
        public int Dimension { get; set; }
        public double[][] Vertexes { get; set; }
        public double[] Values { get; set; }
    }
}
