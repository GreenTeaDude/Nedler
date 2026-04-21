using System;
using NelderMeadMethod.Optimization;

namespace NelderMeadMethod
{
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

            var optimizer = new NelderMeadOptimizer(sphereFunction);

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

//TEST Changge 999
