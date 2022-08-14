from copy import deepcopy
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ipywidgets import VBox, IntSlider, FloatSlider, Dropdown, FloatLogSlider, Checkbox

def configure_plotly_browser_state():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))


def build_2d_grid(xlim, ylim, n_lines=11, n_points=1000):
    """Returns a 2D grid of boundaries given by `xlim` and `ylim`,
     composed of `n_lines` evenly spaced lines of `n_points` each.

    Parameters
    ----------
    xlim : tuple of 2 ints
        Boundaries for the X axis of the grid.
    ylim : tuple of 2 ints
        Boundaries for the Y axis of the grid.
    n_lines : int, optional
        Number of grid lines. Default is 11.
        If n_lines equals n_points, the grid can be used as
        coordinates for the surface of a contourplot.
    n_points: int, optional
        Number of points in each grid line. Default is 1,000.

    Returns
    -------
    lines : ndarray
        For the cases where n_lines is less than n_points, it
        returns an array of shape (2 * n_lines, n_points, 2)
        containing both vertical and horizontal lines of the grid.
        If n_lines equals n_points, it returns an array of shape
        (n_points, n_points, 2), containing all evenly spaced
        points inside the grid boundaries.
    """
    xs = np.linspace(*xlim, num=n_lines)
    ys = np.linspace(*ylim, num=n_points)
    x0, y0 = np.meshgrid(xs, ys)
    lines_x0 = np.atleast_3d(x0.transpose())
    lines_y0 = np.atleast_3d(y0.transpose())

    xs = np.linspace(*xlim, num=n_points)
    ys = np.linspace(*ylim, num=n_lines)
    x1, y1 = np.meshgrid(xs, ys)
    lines_x1 = np.atleast_3d(x1)
    lines_y1 = np.atleast_3d(y1)

    vertical_lines = np.concatenate([lines_x0, lines_y0], axis=2)
    horizontal_lines = np.concatenate([lines_x1, lines_y1], axis=2)

    if n_lines != n_points:
        lines = np.concatenate([vertical_lines, horizontal_lines], axis=0)
    else:
        lines = vertical_lines

    return lines

def data():
    np.random.seed(42)
    orig_x1 = np.linspace(-1, 3, 1000) + np.random.randn(1000)
    orig_x2 = np.linspace(20, 50, 1000) + np.random.randn(1000)
    orig_y = 0 + 2 * orig_x1 + 0.05 * orig_x2 + 2 * np.random.randn(1000)

    shuffled = list(range(1000))
    np.random.shuffle(shuffled)
    x1 = orig_x1[shuffled[:50]].reshape(-1, 1)
    x2 = orig_x2[shuffled[:50]].reshape(-1, 1)
    y = orig_y[shuffled[:50]]
    return x1, x2, y

class plotGradientDescent(object):
    def __init__(self, x1, x2, y):
        self.orig_x1 = x1
        self.orig_x2 = x2
        self.orig_y = y
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.stdsc = StandardScaler(with_mean=True, with_std=True)
        self.linr, self.losses, self.scaled_linr, self.scaled_losses = self.fit()

    def fit(self):
        linr = LinearRegression(fit_intercept=False)
        linr.fit(np.concatenate([self.x1, self.x2], axis=1), self.y)

        scaled_linr = LinearRegression(fit_intercept=False)
        self.scaled_x1, self.scaled_x2 = self.scale()
        scaled_linr.fit(np.concatenate([self.scaled_x1, self.scaled_x2], axis=1), self.y)

        self.grid = build_2d_grid(np.array([-3, 3]) + linr.coef_[0],
                                  np.array([3, -3]) + linr.coef_[1],
                                  n_lines=200, n_points=200)

        self.scaled_grid = build_2d_grid(np.array([-3, 3]) + scaled_linr.coef_[0],
                                  np.array([3, -3]) + scaled_linr.coef_[1],
                                  n_lines=200, n_points=200)

        losses = self.loss_surface(self.grid, self.x1, self.x2, self.y)
        scaled_losses = self.loss_surface(self.scaled_grid, self.scaled_x1, self.scaled_x2, self.y)

        return linr, losses, scaled_linr, scaled_losses

    def scale(self):
        scaled = self.stdsc.fit_transform(np.concatenate([self.orig_x1, self.orig_x2], axis=1))
        x1 = scaled[:, 0, np.newaxis]
        x2 = scaled[:, 1, np.newaxis]
        return x1, x2

    def sort(self):
        xy = np.array([np.array([x1, x2, y]) for x1, x2, y in sorted(zip(self.x1.ravel(),
                                                                         self.x2.ravel(),
                                                                         self.y.ravel()))])
        self.x1 = xy[:, 0]
        self.x2 = xy[:, 1]
        self.y = xy[:, 2].ravel()

        x = np.array([np.array([x1, x2]) for x1, x2 in sorted(zip(self.scaled_x1.ravel(),
                                                                  self.scaled_x2.ravel()))])
        self.scaled_x1 = x[:, 0]
        self.scaled_x2 = x[:, 1]

    def reset_X(self):
        self.x1 = self.orig_x1
        self.x2 = self.orig_x2
        self.y = self.orig_y
        self.scaled_x1, self.scaled_x2 = self.scale()

    def loss_surface(self, grid, x1, x2, y):
        predictions = np.dot(grid, np.concatenate([x1, x2], axis=1).T)
        losses = np.apply_along_axis(func1d=lambda v: mean_squared_error(y, v), axis=2, arr=predictions)
        return losses

    def step(self, x1, x2, y, m1, m2, b, lr=0.0001):
        N = len(y)
        y_current = (m1 * x1 + m2 * x2) + b
        error = np.array(y - y_current)
        cost = (error ** 2).mean()
        m1_gradient = -(2/N) * sum(x1 * error)
        m2_gradient = -(2/N) * sum(x2 * error)
        b_gradient = -(2/N) * sum(error)
        m1 = m1 - (lr * m1_gradient)
        m2 = m2 - (lr * m2_gradient)
        b = b - (lr * b_gradient)
        return m1, m2, b, cost

    def train(self, m1=0, m2=0, b=0, batch_size=None, epochs=1000, lr=0.0001, scaled=True):
        if scaled:
            grid = self.scaled_grid
            losses = self.scaled_losses
            x1 = self.scaled_x1.ravel()
            x2 = self.scaled_x2.ravel()
            m1min = self.scaled_linr.coef_[0]
            m2min = self.scaled_linr.coef_[1]
        else:
            grid = self.grid
            losses = self.losses
            x1 = self.x1.ravel()
            x2 = self.x2.ravel()
            m1min = self.linr.coef_[0]
            m2min = self.linr.coef_[1]
        y = self.y.ravel()

        N = len(y)
        if (batch_size is None) or (batch_size > N):
            batch_size = N
        n_batches = int(N // batch_size)

        m1_history = [m1]
        m2_history = [m2]
        b_history = [b]
        y_current = (m1 * x1 + m2 * x2) + b
        error = np.array(y - y_current)
        cost = (error ** 2).mean()
        cost_history = [cost]
        for i in range(epochs):
            for j in range(n_batches):
                clause = slice(j * batch_size, (j + 1) * batch_size)
                m1, m2, b, cost = self.step(x1[clause],
                                            x2[clause],
                                            y[clause],
                                            m1, m2, b, lr)
                m1_history.append(m1)
                m2_history.append(m2)
                b_history.append(b)
                cost_history.append(cost)

        self.contour = self.plot_contour(grid, losses.T)
        self.path = self.plot_path(m1_history, m2_history)
        self.minimum = self.plot_minimum(m1min, m2min)

    @property
    def traces(self):
        return dict(contour=self.contour, path=self.path, minimum=self.minimum)

    def plot_contour(self, grid, losses):
        contour = go.Contour(x=grid[:, 0, 0],
                             y=grid[0, :, 1],
                             z=losses)
        return contour

    def plot_path(self, m1, m2):
        path = go.Scatter(x=m1, y=m2, mode='lines')
        return path

    def plot_minimum(self, m1, m2):
        minimum = go.Scatter(x=[m1], y=[m2], mode='markers', marker={'symbol': 'star'})
        return minimum

def build_figure(gd_obj):
    fig = make_subplots(1, 1, print_grid=False)
    fig.append_trace(go.Contour(x=gd_obj.scaled_grid[:, 0, 0],
                                 y=gd_obj.scaled_grid[0, :, 1],
                                 z=gd_obj.scaled_losses), 1, 1)
    fig.append_trace(go.Scatter(x=[], y=[], mode='lines'), 1, 1)
    fig.append_trace(go.Scatter(x=[gd_obj.scaled_linr.coef_[0]],
                                y=[gd_obj.scaled_linr.coef_[1]],
                                mode='markers', marker={'symbol': 'star'}), 1, 1)

    #f = go.FigureWidget(fig)
    f = go.Figure(fig)

    f['layout'].update(title='Gradient Descent')
    f['layout']['xaxis'].update(range=np.array([-3, 3]) + gd_obj.scaled_linr.coef_[0])
    f['layout']['yaxis'].update(range=np.array([-3, 3]) + gd_obj.scaled_linr.coef_[1])
    f['layout']['autosize'] = False
    f['layout']['width'] = 600
    f['layout']['height'] = 600
    f['layout']['showlegend'] = False

    names = ['contour', 'path', 'minimum']

    m1range = (np.array([-3, 3]) + gd_obj.scaled_linr.coef_[0])
    m2range = (np.array([-3, 3]) + gd_obj.scaled_linr.coef_[1])
    m1range = [np.ceil(m1range[0]), np.floor(m1range[1])]
    m2range = [np.ceil(m2range[0]), np.floor(m2range[1])]

    def update(lr, scaled, epochs, batch_size, m1, m2):
        gd_obj.train(m1, m2, 0, batch_size=int(batch_size), epochs=epochs, lr=lr, scaled=scaled)
        values = {'contour': {'x': None, 'y': None, 'z': gd_obj.contour.z},
                  'path': {'x': gd_obj.path.x, 'y': gd_obj.path.y},
                  'minimum': {'x': gd_obj.minimum.x, 'y': gd_obj.minimum.y}}
        with f.batch_update():
            for i, data in enumerate(f.data):
                try:
                    if values[names[i]]['z'] is not None:
                        data.z = values[names[i]]['z']
                except KeyError:
                    pass
                if values[names[i]]['y'] is not None:
                    data.y = values[names[i]]['y']
                if values[names[i]]['x'] is not None:
                    data.x = values[names[i]]['x']

    lr = FloatLogSlider(description='Learning Rate', value=0.0001, base=10, min=-4, max=-.5, step=.25)
    scaled = Checkbox(description='Scale Features', value=False)
    epochs = IntSlider(description='Epochs', value=100, min=100, max=500, step=100)
    batch_size = FloatLogSlider(description='Batch Size', value=16, base=2, min=0, max=6, step=1)
    m1 = IntSlider(description='x1', value=1, min=m1range[0], max=m1range[1], step=1)
    m2 = IntSlider(description='x2', value=1, min=m2range[0], max=m2range[1], step=1)

    #return (f, interactive(update, lr=lr, scaled=scaled, epochs=epochs, batch_size=batch_size, m1=m1, m2=m2))
    return f, update, (lr, scaled, epochs, batch_size, m1, m2)
