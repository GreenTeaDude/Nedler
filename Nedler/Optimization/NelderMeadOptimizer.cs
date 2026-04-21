using System;
using NelderMeadMethod.Core;

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

            for (int iter = 0; iter < maxIterations; iter++)
            {
                if (simplex.IsConverged(Accuracy))
                    break;

                var centroid = simplex.CalculateCentroid();
                var worst = simplex.GetWorstVertex();

                double best = simplex.GetValue(0);
                double secondWorst = simplex.GetSecondWorstValue();
                double worstVal = simplex.GetValue(simplex.VertexCount - 1);

                // Reflection
                double[] reflected = new double[simplex.Dimension];
                for (int i = 0; i < simplex.Dimension; i++)
                    reflected[i] = centroid[i] + ReflectionCoeff * (centroid[i] - worst[i]);

                double reflectedVal = objectiveFunction(reflected);

                if (reflectedVal < best)
                {
                    // Expansion
                    double[] expanded = new double[simplex.Dimension];
                    for (int i = 0; i < simplex.Dimension; i++)
                        expanded[i] = centroid[i] + ExpansionCoeff * (reflected[i] - centroid[i]);

                    double expandedVal = objectiveFunction(expanded);

                    if (expandedVal < reflectedVal)
                        simplex.ReplaceWorstVertex(expanded, expandedVal);
                    else
                        simplex.ReplaceWorstVertex(reflected, reflectedVal);

                    simplex.SortVertexes();
                    continue;
                }

                if (reflectedVal < secondWorst)
                {
                    simplex.ReplaceWorstVertex(reflected, reflectedVal);
                    simplex.SortVertexes();
                    continue;
                }

                // Contraction
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
                    continue;
                }

                // Shrink
                simplex.Shrink(objectiveFunction, ShrinkCoeff);
            }

            return simplex.GetBestVertex();
        }
    }
}