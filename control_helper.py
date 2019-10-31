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
        return np.array([dx_dt, dpx_dt, dtheta_dt, dptheta_dt]) # = dy/dt

    def rhs_forced(self, t, y_forced):
        ''' Probably will want some time dependence for the force '''
        y, force = y_forced
        return self.rhs(t,y) + np.array([0, force, 0, 0])

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
        return np.array([0,force,0,0]).reshape(-1,1)


# -----------------------------------------------------
# ------------------ LQR solver -----------------------
# -----------------------------------------------------

import scipy.linalg
# shout-out to http://www.kostasalexis.com/lqr-control.html
# just need to know the continuous ricatti equation solver exists
 
def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151
 
    #first, try to solve the ricatti equation
    X = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.array(scipy.linalg.inv(R)@(B.T@X))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B@K)
     
    return K, X, eigVals
 
def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
     
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
 
    #first, try to solve the ricatti equation
    X = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.array(scipy.linalg.inv(B.T@X@B+R)*(B.T@X@A))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B@K)
     
    return K, X, eigVals

# ------------------------------------------------------
# ------------------ Visual utilities ------------------
# ------------------------------------------------------

# Imports for array display
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display

# Imports for figure display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import color

# Import for animation
from matplotlib import animation

class movie_maker:
    '''
    Usage: HTML(m.animate(2).to_jshtml())
    '''
    def __init__(self, cartpole, x, th, times):
        # object and coordiantes to animate
        self.c = cartpole
        self.x = x
        self.th = th
        self.time = times    
        
        # Create figure
        self.fig, self.ax = plt.subplots(1)

        xmin,xmax = [min(self.x)-2*self.c.l, max(self.x)+2*self.c.l]
        ymin,ymax = [-2*self.c.l, 2*self.c.l]

        self.ax.set_xlim([xmin,xmax])
        self.ax.set_ylim([ymin,ymax])
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        self.time_text = self.ax.text(0.02, 0.9, '', fontsize=20, transform=self.ax.transAxes)
        self.path, = self.ax.plot([], [], ls='--', lw=1)
        self.pole, = self.ax.plot([],[],  c='k')
        self.cart, = self.ax.plot([],[], marker='s', ms=20, c='k')
        self.mass, = self.ax.plot([],[], marker='o', c='k')

    def _init(self):
        self.path.set_data([],[])
        self.pole.set_data([],[])
        self.cart.set_data([],[])
        self.mass.set_data([],[])
        self.time_text.set_text('')
        return self.path,self.pole,self.cart,self.mass,self.time_text

    def _animate(self, i):
        # Assert bounds
        assert (i < len(self.time)), "No frames at index i"

        # Pick data
        cm_pole = [self.x[i]+self.c.l*np.sin(self.th[i]), self.c.l*np.cos(self.th[i])]
        path_pole = [self.x[:i]+self.c.l*np.sin(self.th[:i]), self.c.l*np.cos(self.th[:i])]
        cm_cart = [self.x[i], 0]

        # Update data
        self.path.set_data(*path_pole)
        self.pole.set_data([cm_cart[0], 2*(cm_pole[0]-cm_cart[0])+cm_cart[0]],
                      [cm_cart[1], 2*cm_pole[1]])
        self.cart.set_data(*cm_cart)
        self.mass.set_data(*cm_pole)
        self.time_text.set_text(str(round(self.time[i], 2)))
        return self.path,self.pole,self.cart,self.mass,self.time_text

    def animate(self, interval = 1):
        ani = animation.FuncAnimation(self.fig, self._animate, frames=len(self.time),
                                      interval=interval, blit=True, init_func=self._init)
        # don't show plot if %inline
        plt.close();
        return ani

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

def make_movie_with_buffer(ctpl, x, th, time, tdiv=10):
    '''
    Make a movie of the x, th coordintes over time by flushing
    the images from the buffer to the screen.

    parameters:
        ctpl: cartpole object
        x: x coordinates
        th: theta coordinates
        time: list of times
        tdiv: print a frame every tdiv time-steps

    All of x, th, and time need to have the same length. ctpl is used 
    to get the physical constants of the equations.
    '''
    xmin,xmax = [min(x)-2*ctpl.l, max(x)+2*ctpl.l]
    ymin,ymax = [-2*ctpl.l, 2*ctpl.l]
    for i,t in enumerate(time):
        if i%tdiv==0:
            fig, ax = plt.subplots(1)
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            ax.set_aspect('equal')

            # Pick data
            cm_pole = [x[i]+ctpl.l*np.sin(th[i]), ctpl.l*np.cos(th[i])]
            ax.plot(x[:i]+ctpl.l*np.sin(th[:i]), ctpl.l*np.cos(th[:i]), ls='--', lw=1)
            cm_cart = [x[i], 0]
            
            # Plot data
            ax.set_title(str(round(t, 2)), fontsize=20)
            ax.plot([cm_cart[0], 2*(cm_pole[0]-cm_cart[0])+cm_cart[0]], [cm_cart[1], 2*cm_pole[1]], c='k')
            ax.scatter(*cm_pole, marker='o', c='k')
            ax.scatter(*cm_cart, marker='s', s=500, c='k')
            ax.axis('off')

            plt.close() # prevent print
            DisplayFigure(fig)