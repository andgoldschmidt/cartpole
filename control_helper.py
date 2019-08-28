# Functions for control theory notebook
import numpy as np

# ------------------------------------------------------
# ------------------ Cartpole object ------------------
# -----------------------------------------------------
class cartpole:
    def __init__(self, params):
        '''
        The state of the dynamics system is y = [x, px, theta, ptheta]

        params:
            mass_cart: M
            mass_pole: m
            length_pole: equivalently, a rigid pendulum of mass m at l/2
            friction: eta
            
        other:
            g: gravitational constant
        '''
        try:
            self.M = params['mass_cart']
            self.m = params['mass_pole']    
            self.l = params['length_pole']/2 # center of mass is halfway along the pole
        except:
            print('Error initializing cartpole instance: Missing required parameter.')
        
        # Check for dissipation, gravity, and noise in params   
        self.eta = params['friction'] if 'friction' in params.keys() else 0
        self.g = params['gravity'] if 'gravity' in params.keys() else 10 # choose units
            
    def rhs(self, t, y):
        '''
          Analytic solution to the 1st order Hamiltonian equations of motion;
        the right hand side of dY/dt = f(Y,t)
        '''
        # no explicit time dependence
        x, px, theta, ptheta = y
        cosX = np.cos(theta)
        sinX = np.sin(theta)
        D = (self.m*self.l**2)*(self.M + self.m*sinX**2)
        Dinv = (1/D)
        dx_dt = Dinv*(-self.m*self.l*cosX*ptheta + self.m*self.l**2*px)
        dpx_dt = -self.eta*dx_dt
        dtheta_dt = Dinv*(-self.m*self.l*cosX*px + (self.M+self.m)*ptheta)
        dptheta_dt = Dinv**2*(self.m**2*self.l**2*cosX*sinX) \
                     *((self.M+self.m)*ptheta**2 + self.m*self.l**2*px**2 - 2*self.m*self.l*cosX*px*ptheta) \
                     + self.m*self.g*self.l*sinX
        return [dx_dt, dpx_dt, dtheta_dt, dptheta_dt] # = dy/dt

    def rhs_forced(self, t, y_forced):
        y, force = y_forced
        return [i+j for i,j in zip(self.rhs(t,y), [0, force, 0, 0])]

    def A(self, s):
        '''
        0:  s = 1
        pi: s = -1
        '''
        return np.array([
            [0, 1/self.M, 0, -s/(self.M*self.l)],
            [0, -self.eta/self.M, 0, s*self.eta/(self.l*self.M)],
            [0, -s/(self.M*self.l), 0, (self.M+self.m)/(self.M*self.m*self.l**2)],
            [0, 0, s*self.m*self.g*self.l, 0]])

    def B(self, force):
        return [0,force,0,0]
# ------------------------------------------------------
# ------------------ Visual utilities ------------------
# ------------------------------------------------------
# Imports for array display
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display

# Imports for figure display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import color

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  clear_output(wait = True)
  display(Image(data=f.getvalue()))

def DisplayFigure(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    DisplayArray(color.rgb2gray(X))