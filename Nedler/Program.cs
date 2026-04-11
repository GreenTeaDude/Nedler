using System;
using System.Linq;

namespace NelderMeadMethod
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
                throw new ArgumentOutOfRangeException(nameof(step), "Шаг должен быть положительным.");

            Dimension = startPoint.Length;
            VertexCount = Dimension + 1;

            Vertexes = new double[VertexCount][];
            Values = new double[VertexCount];

           
            Vertexes[0] = new double[Dimension];
            Array.Copy(startPoint, Vertexes[0], Dimension);

            for (int i = 1; i < VertexCount; i++)
            {
                Vertexes[i] = new double[Dimension];
                Array.Copy(startPoint, Vertexes[i], Dimension);
                Vertexes[i][i - 1] += step;
            }
        }

        public void UpdateValues(Func<double[], double> objectiveFunction)
        {
            if (objectiveFunction == null)
                throw new ArgumentNullException(nameof(objectiveFunction));

            for (int i = 0; i < VertexCount; i++)
            {
                Values[i] = objectiveFunction(Vertexes[i]);
            }
        }

        public double GetValue(int vertexIndex) => Values[vertexIndex];

        public double[] GetWorstVertex()
        {
            double[] result = new double[Dimension];
            Array.Copy(Vertexes[VertexCount - 1], result, Dimension);
            return result;
        }

        public double[] GetBestVertex()
        {
            double[] result = new double[Dimension];
            Array.Copy(Vertexes[0], result, Dimension);
            return result;
        }

        public double GetSecondWorstValue() => Values[VertexCount - 2];

        public void ReplaceWorstVertex(double[] newVertex, double newValue)
        {
            if (newVertex == null)
                throw new ArgumentNullException(nameof(newVertex));

            if (newVertex.Length != Dimension)
                throw new ArgumentException("Размерность вершины не совпадает с размерностью симплекса.", nameof(newVertex));

            Array.Copy(newVertex, Vertexes[VertexCount - 1], Dimension);
            Values[VertexCount - 1] = newValue;
        }

        public double[] CalculateCentroid()
        {
            double[] centroid = new double[Dimension];

            for (int i = 0; i < VertexCount - 1; i++)
            {
                for (int j = 0; j < Dimension; j++)
                {
                    centroid[j] += Vertexes[i][j];
                }
            }

            for (int j = 0; j < Dimension; j++)
            {
                centroid[j] /= (VertexCount - 1);
            }

            return centroid;
        }

        public bool IsConverged(double accuracy)
        {
            if (accuracy <= 0)
                throw new ArgumentOutOfRangeException(nameof(accuracy), "Точность должна быть положительной.");

            return Math.Abs(Values[VertexCount - 1] - Values[0]) < accuracy;
        }

        public void SortVertexes()
        {
            var indexedValues = Values
                .Select((value, index) => new { Value = value, Index = index })
                .OrderBy(x => x.Value)
                .ToList();

            double[][] sortedVertexes = new double[VertexCount][];
            double[] sortedValues = new double[VertexCount];

            for (int i = 0; i < VertexCount; i++)
            {
                int originalIndex = indexedValues[i].Index;
                sortedVertexes[i] = Vertexes[originalIndex];
                sortedValues[i] = Values[originalIndex];
            }

            Vertexes = sortedVertexes;
            Values = sortedValues;
        }

        public void Shrink(Func<double[], double> objectiveFunction, double sigma = 0.5)
        {
            if (objectiveFunction == null)
                throw new ArgumentNullException(nameof(objectiveFunction));

            if (sigma <= 0 || sigma >= 1)
                throw new ArgumentOutOfRangeException(nameof(sigma), "Коэффициент shrink должен быть в диапазоне (0, 1).");

            double[] best = GetBestVertex();

            for (int i = 1; i < VertexCount; i++)
            {
                for (int j = 0; j < Dimension; j++)
                {
                    Vertexes[i][j] = best[j] + sigma * (Vertexes[i][j] - best[j]);
                }

                Values[i] = objectiveFunction(Vertexes[i]);
            }

            Values[0] = objectiveFunction(Vertexes[0]);
            SortVertexes();
        }
    }

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
            if (startPoint == null)
                throw new ArgumentNullException(nameof(startPoint));

            if (startPoint.Length == 0)
                throw new ArgumentException("Начальная точка не должна быть пустой.", nameof(startPoint));

            if (maxIterations <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxIterations), "Число итераций должно быть положительным.");

            if (Step <= 0)
                throw new ArgumentOutOfRangeException(nameof(Step), "Шаг должен быть положительным.");

            if (Accuracy <= 0)
                throw new ArgumentOutOfRangeException(nameof(Accuracy), "Точность должна быть положительной.");

            Simplex simplex = new Simplex(startPoint, Step);
            simplex.UpdateValues(objectiveFunction);
            simplex.SortVertexes();

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                if (simplex.IsConverged(Accuracy))
                    break;

                double[] centroid = simplex.CalculateCentroid();
                double[] worst = simplex.GetWorstVertex();

                double bestValue = simplex.GetValue(0);
                double secondWorstValue = simplex.GetSecondWorstValue();
                double worstValue = simplex.GetValue(simplex.VertexCount - 1);

              
                double[] reflected = new double[simplex.Dimension];
                for (int j = 0; j < simplex.Dimension; j++)
                {
                    reflected[j] = centroid[j] + ReflectionCoeff * (centroid[j] - worst[j]);
                }

                double reflectedValue = objectiveFunction(reflected);

               
                if (reflectedValue < bestValue)
                {
                    double[] expanded = new double[simplex.Dimension];
                    for (int j = 0; j < simplex.Dimension; j++)
                    {
                        expanded[j] = centroid[j] + ExpansionCoeff * (reflected[j] - centroid[j]);
                    }

                    double expandedValue = objectiveFunction(expanded);

                    if (expandedValue < reflectedValue)
                        simplex.ReplaceWorstVertex(expanded, expandedValue);
                    else
                        simplex.ReplaceWorstVertex(reflected, reflectedValue);

                    simplex.SortVertexes();
                    continue;
                }

              
                if (reflectedValue < secondWorstValue)
                {
                    simplex.ReplaceWorstVertex(reflected, reflectedValue);
                    simplex.SortVertexes();
                    continue;
                }

             
                double[] contracted = new double[simplex.Dimension];

                
                if (reflectedValue < worstValue)
                {
                    for (int j = 0; j < simplex.Dimension; j++)
                    {
                        contracted[j] = centroid[j] + ContractionCoeff * (reflected[j] - centroid[j]);
                    }
                }
               
                else
                {
                    for (int j = 0; j < simplex.Dimension; j++)
                    {
                        contracted[j] = centroid[j] - ContractionCoeff * (centroid[j] - worst[j]);
                    }
                }

                double contractedValue = objectiveFunction(contracted);

                if (contractedValue < worstValue)
                {
                    simplex.ReplaceWorstVertex(contracted, contractedValue);
                    simplex.SortVertexes();
                    continue;
                }

              
                simplex.Shrink(objectiveFunction, ShrinkCoeff);
            }

            return simplex.GetBestVertex();
        }
    }

    internal class Program
    {
        static void Main(string[] args)
        {
            Func<double[], double> sphereFunction = coords =>
            {
                double sum = 0;
                for (int i = 0; i < coords.Length; i++)
                {
                    sum += coords[i] * coords[i];
                }
                return sum;
            };

            double[] start = { 1.0, 1.5, 1.0 };

            NelderMeadOptimizer optimizer = new NelderMeadOptimizer(sphereFunction)
            {
                Step = 0.5,
                Accuracy = 1e-6,
                ReflectionCoeff = 1.0,
                ExpansionCoeff = 2.0,
                ContractionCoeff = 0.5,
                ShrinkCoeff = 0.5
            };

            double[] result = optimizer.Optimize(start, 500);

            Console.WriteLine("Найденный минимум:");
            for (int i = 0; i < result.Length; i++)
            {
                Console.WriteLine($"x{i + 1} = {result[i]:F6}");
            }

            Console.WriteLine($"Значение функции = {sphereFunction(result):F6}");
        }
    }
}
