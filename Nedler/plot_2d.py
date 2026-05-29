import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.colors as colors

def read_simplex_history(filename):
    """Чтение истории эволюции симплекса из CSV файла"""
    data = pd.read_csv(filename, comment='#', header=None)
    data.columns = ['iteration', 'vertex_id', 'x1', 'x2', 'value']
    
    # Получаем уникальные итерации
    iterations = sorted(data['iteration'].unique())
    
    simplex_history = []
    for iter_num in iterations:
        iter_data = data[data['iteration'] == iter_num]
        vertexes = iter_data[['x1', 'x2']].values
        values = iter_data['value'].values
        
        # Порядок вершин для построения треугольника
        # Для 2D симплекса - это треугольник
        if len(vertexes) == 3:  # Для 2D
            simplex_history.append({
                'iteration': iter_num,
                'vertexes': vertexes,
                'values': values,
                'best_point': vertexes[np.argmin(values)],
                'best_value': np.min(values),
                'worst_point': vertexes[np.argmax(values)],
                'worst_value': np.max(values)
            })
    
    return simplex_history

def create_contour_plot(func_str, bounds, resolution=100):
    """Создание контурного графика целевой функции"""
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Вычисляем значения функции
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            try:
                # Создаём функцию из строки
                # ВНИМАНИЕ: Это упрощённая версия, может потребоваться доработка
                expr = func_str.replace('x0', f'({X[i,j]})').replace('x1', f'({Y[i,j]})')
                expr = expr.replace('^', '**')
                # Используем eval (осторожно! только для доверенных выражений)
                Z[i,j] = eval(expr)
            except:
                Z[i,j] = np.nan
    
    return X, Y, Z

class SimplexAnimator:
    def __init__(self, simplex_history, func_str=None, bounds=None):
        self.simplex_history = simplex_history
        self.func_str = func_str
        self.bounds = bounds
        self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 6))
        
    def animate(self, frame):
        """Анимация одного кадра"""
        # Очищаем оси
        for ax in self.ax:
            ax.clear()
        
        # Получаем данные для текущего кадра
        if frame >= len(self.simplex_history):
            frame = len(self.simplex_history) - 1
        
        current = self.simplex_history[frame]
        
        # Левый график: эволюция симплекса
        self.ax[0].set_title(f'Simplex Evolution - Iteration {current["iteration"]}')
        
        # Рисуем контурный график, если есть функция и границы
        if self.func_str and self.bounds:
            X, Y, Z = create_contour_plot(self.func_str, self.bounds, resolution=50)
            contour = self.ax[0].contourf(X, Y, Z, levels=20, alpha=0.6, cmap='viridis')
            self.ax[0].contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
            plt.colorbar(contour, ax=self.ax[0], label='Function Value')
        
        # Рисуем треугольник симплекса
        vertexes = current['vertexes']
        triangle = np.vstack([vertexes, vertexes[0]])  # Замыкаем треугольник
        
        # Цвет вершин в зависимости от значения функции
        colors_vertices = ['red' if v == current['best_value'] else 
                          'blue' if v == current['worst_value'] else 
                          'green' for v in current['values']]
        
        # Рисуем стороны треугольника
        self.ax[0].plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=2, alpha=0.7)
        
        # Рисуем вершины
        for i, (x, y) in enumerate(vertexes):
            self.ax[0].plot(x, y, 'o', color=colors_vertices[i], markersize=8, 
                           label=f'Vertex {i}' if frame == 0 else "")
            # Добавляем аннотацию со значением
            self.ax[0].annotate(f'{current["values"][i]:.2f}', 
                               (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        # Отмечаем лучшую точку
        best = current['best_point']
        self.ax[0].plot(best[0], best[1], '*', color='gold', markersize=15, 
                       label='Best point' if frame == 0 else "")
        
        # Настройки графика
        if self.bounds:
            self.ax[0].set_xlim(self.bounds[0])
            self.ax[0].set_ylim(self.bounds[1])
        else:
            # Автоматические границы с отступом
            margin = 0.2
            x_min, x_max = vertexes[:, 0].min(), vertexes[:, 0].max()
            y_min, y_max = vertexes[:, 1].min(), vertexes[:, 1].max()
            dx = (x_max - x_min) * margin
            dy = (y_max - y_min) * margin
            self.ax[0].set_xlim(x_min - dx, x_max + dx)
            self.ax[0].set_ylim(y_min - dy, y_max + dy)
        
        self.ax[0].set_xlabel('x0')
        self.ax[0].set_ylabel('x1')
        self.ax[0].grid(True, alpha=0.3)
        self.ax[0].legend(loc='upper right')
        
        # Правый график: сходимость
        iterations = [s['iteration'] for s in self.simplex_history[:frame+1]]
        best_values = [s['best_value'] for s in self.simplex_history[:frame+1]]
        
        self.ax[1].plot(iterations, best_values, 'g-', linewidth=2, label='Best value')
        self.ax[1].set_xlabel('Iteration')
        self.ax[1].set_ylabel('Function Value')
        self.ax[1].set_title('Convergence')
        self.ax[1].set_yscale('log')
        self.ax[1].grid(True, alpha=0.3)
        self.ax[1].legend()
        
        # Отмечаем текущую итерацию
        if len(iterations) > 1:
            self.ax[1].axvline(x=current['iteration'], color='red', 
                              linestyle='--', alpha=0.5, label='Current')
        
        self.fig.suptitle(f'Nelder-Mead Simplex Optimization\nIteration: {current["iteration"]}', 
                         fontsize=14, fontweight='bold')
        
        return self.ax

def create_static_visualization(simplex_history, func_str=None, bounds=None):
    """Создание статической визуализации с несколькими кадрами"""
    n_frames = min(9, len(simplex_history))
    frames_idx = np.linspace(0, len(simplex_history)-1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(frames_idx):
            frame_idx = frames_idx[idx]
            current = simplex_history[frame_idx]
            
            ax.set_title(f'Iteration {current["iteration"]}')
            
            # Рисуем контуры
            if func_str and bounds:
                X, Y, Z = create_contour_plot(func_str, bounds, resolution=50)
                ax.contourf(X, Y, Z, levels=20, alpha=0.6, cmap='viridis')
            
            # Рисуем симплекс
            vertexes = current['vertexes']
            triangle = np.vstack([vertexes, vertexes[0]])
            ax.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=2, alpha=0.7)
            ax.plot(vertexes[:, 0], vertexes[:, 1], 'ro', markersize=6)
            
            # Отмечаем лучшую точку
            best = current['best_point']
            ax.plot(best[0], best[1], '*', color='gold', markersize=12)
            
            # Настройки
            if bounds:
                ax.set_xlim(bounds[0])
                ax.set_ylim(bounds[1])
            ax.set_xlabel('x0')
            ax.set_ylabel('x1')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Читаем данные
    try:
        simplex_history = read_simplex_history('simplex_history.csv')
        print(f"Загружено {len(simplex_history)} итераций")
        
        # Читаем функцию из файла (опционально)
        func_str = None
        try:
            with open('function.txt', 'r') as f:
                func_str = f.read().strip()
        except:
            pass
        
        # Границы для визуализации (можно задать вручную)
        bounds = [[-2, 2], [-2, 2]]  # x0 и x1 границы
        
        # Создаём анимацию
        animator = SimplexAnimator(simplex_history, func_str, bounds)
        
        # Создаём анимацию
        anim = FuncAnimation(animator.fig, animator.animate, 
                           frames=len(simplex_history), 
                           interval=500,  # 500ms между кадрами
                           repeat=True,
                           blit=False)
        
        # Сохраняем анимацию
        anim.save('simplex_animation.gif', writer='pillow', fps=2)
        print("Анимация сохранена как 'simplex_animation.gif'")
        
        # Создаём статическую визуализацию
        fig_static = create_static_visualization(simplex_history, func_str, bounds)
        fig_static.savefig('simplex_evolution.png', dpi=150, bbox_inches='tight')
        print("Статическая визуализация сохранена как 'simplex_evolution.png'")
        
        # Показываем анимацию
        plt.show()
        
    except FileNotFoundError:
        print("Ошибка: файл simplex_history.csv не найден")
        print("Убедитесь, что C# программа успешно завершилась и создала файл")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
