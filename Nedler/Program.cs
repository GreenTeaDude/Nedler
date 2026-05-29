using System;
using NelderMeadMethod.Optimization;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace NelderMeadMethod
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== Метод оптимизации Нелдера-Мида ===");
            Console.WriteLine();

            Func<double[], double> userFunction = GetUserFunction();

            
            Console.WriteLine("\nПримечание: Для визуализации симплекса в 2D используйте функцию с 2 переменными (x0, x1)");
            Console.WriteLine("Если размерность > 2, будет сохранена проекция на первые две координаты\n");

            double[] start = GetStartPoint();

            Console.WriteLine("\n--- Настройка параметров оптимизации ---");
            Console.Write("Введите максимальное количество итераций (по умолчанию 500): ");
            string maxIterInput = Console.ReadLine();
            int maxIterations = string.IsNullOrWhiteSpace(maxIterInput) ? 500 : int.Parse(maxIterInput);

            var optimizer = new NelderMeadOptimizer(userFunction);

            Console.Write("Введите точность (по умолчанию 1e-6): ");
            string accuracyInput = Console.ReadLine();
            if (!string.IsNullOrWhiteSpace(accuracyInput))
            {
                optimizer.Accuracy = double.Parse(accuracyInput, CultureInfo.InvariantCulture);
            }

          
            Console.Write("Введите шаг для построения начального симплекса (по умолчанию 0.5): ");
            string stepInput = Console.ReadLine();
            if (!string.IsNullOrWhiteSpace(stepInput))
            {
                optimizer.Step = double.Parse(stepInput, CultureInfo.InvariantCulture);
            }

            Console.WriteLine("\nОптимизация запущена...");
            Stopwatch stopwatch = Stopwatch.StartNew();

            double[] result = optimizer.Optimize(start, maxIterations);
            stopwatch.Stop();

            
            var lines = optimizer.HistoryValues
                .Select((value, index) => $"{index},{value.ToString(CultureInfo.InvariantCulture)}");
            File.WriteAllLines("convergence.csv", lines);

           
            optimizer.SaveSimplexHistoryToFile("simplex_history.csv");
            Console.WriteLine($"Сохранена история симплекса ({optimizer.HistorySimplex.Count} итераций)");

           
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = "-3 plot_2d.py",
                    UseShellExecute = true
                };
                Process.Start(psi);
                Console.WriteLine("\nЗапущена визуализация симплекса...");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nНе удалось запустить Python: {ex.Message}");
                Console.WriteLine("Убедитесь, что Python установлен и файл plot_2d.py находится в той же папке");
            }

            PrintResults(userFunction, result, optimizer, stopwatch);

            Console.WriteLine("\nНажмите любую клавишу для выхода...");
            Console.ReadKey();
        }

        static Func<double[], double> GetUserFunction()
        {
            Console.WriteLine("--- Ввод целевой функции ---");
            Console.WriteLine("Доступные операции: +, -, *, /, ^");
            Console.WriteLine("Доступные функции: sin, cos, tan, sqrt, exp, log, abs");
            Console.WriteLine("Переменные: x0, x1, x2, ...");
            Console.WriteLine();
            Console.WriteLine("Примеры 2D функций:");
            Console.WriteLine("  x0^2 + x1^2                     (сфера)");
            Console.WriteLine("  x0^2 + 2*x1^2                  (эллипсоид)");
            Console.WriteLine("  x0^2 - x0*x1 + x1^2 + 10       (квадратичная)");
            Console.WriteLine("  sin(x0) + cos(x1) + x0^2/10    (тригонометрическая)");
            Console.WriteLine("  (1-x0)^2 + 100*(x1-x0^2)^2     (функция Розенброка)");
            Console.WriteLine();

            while (true)
            {
                Console.Write("Введите выражение функции: ");
                string expression = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(expression))
                {
                    Console.WriteLine("Функция не может быть пустой.");
                    continue;
                }

                int dimension = GetDimensionFromExpression(expression);

                try
                {
                    Func<double[], double> func = CreateFunction(expression, dimension);
                    double[] testPoint = new double[dimension];
                    double testValue = func(testPoint);

                    Console.WriteLine($"\nФункция создана! Размерность: {dimension}");
                    Console.WriteLine($"Значение в нуле: {testValue:F6}");
                    Console.WriteLine();
                    return func;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ошибка: {ex.Message}\n");
                }
            }
        }

        static Func<double[], double> CreateFunction(string expression, int dimension)
        {
            
            return (double[] x) =>
            {
                string expr = expression;

                
                for (int i = 0; i < dimension; i++)
                {
                    expr = expr.Replace($"x{i}", x[i].ToString("R", CultureInfo.InvariantCulture));
                }

               
                expr = expr.Replace("^", "**");

                
                expr = expr.Replace("sin", "Math.Sin");
                expr = expr.Replace("cos", "Math.Cos");
                expr = expr.Replace("tan", "Math.Tan");
                expr = expr.Replace("sqrt", "Math.Sqrt");
                expr = expr.Replace("exp", "Math.Exp");
                expr = expr.Replace("log", "Math.Log");
                expr = expr.Replace("abs", "Math.Abs");

                
                var dt = new System.Data.DataTable();
                try
                {
                    
                    var result = dt.Compute(expr, "");
                    return Convert.ToDouble(result);
                }
                catch
                {
                    
                    return double.MaxValue;
                }
            };
        }

        static int GetDimensionFromExpression(string expression)
        {
            var matches = Regex.Matches(expression, @"x(\d+)");
            if (matches.Count == 0)
                return 2;

            int maxIndex = matches.Max(m => int.Parse(m.Groups[1].Value));
            return maxIndex + 1;
        }

        static double[] GetStartPoint()
        {
            Console.WriteLine("--- Ввод начальной точки ---");

            Console.Write("Введите размерность пространства (рекомендуется 2 для визуализации): ");
            int dimension = int.Parse(Console.ReadLine());

            double[] startPoint = new double[dimension];

            for (int i = 0; i < dimension; i++)
            {
                double defaultValue = (i == 0) ? 1.0 : (i == 1) ? 1.0 : 0.0;
                Console.Write($"Введите x{i} (по умолчанию {defaultValue}): ");
                string input = Console.ReadLine();
                startPoint[i] = string.IsNullOrWhiteSpace(input) ? defaultValue : double.Parse(input, CultureInfo.InvariantCulture);
            }

            Console.WriteLine("\nНачальная точка:");
            for (int i = 0; i < dimension; i++)
            {
                Console.WriteLine($"  x{i} = {startPoint[i]:F4}");
            }

            return startPoint;
        }

        static void PrintResults(Func<double[], double> func, double[] result, NelderMeadOptimizer optimizer, Stopwatch sw)
        {
            Console.WriteLine($"\n=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===");
            Console.WriteLine($"Время: {sw.ElapsedMilliseconds} мс");
            Console.WriteLine($"Итераций: {optimizer.HistoryValues.Count}");
            Console.WriteLine("\nКоординаты минимума:");
            for (int i = 0; i < result.Length; i++)
            {
                Console.WriteLine($"  x{i + 1} = {result[i]:F10}");
            }
            Console.WriteLine($"Значение функции: {func(result):F10}");
        }
    }
}
