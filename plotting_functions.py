
import matplotlib.pyplot as plt

class plotDynamic:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'r-', label='Ball')
        self.line2, = self.ax.plot([], [], 'b-', label='Paddle')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Height')
        plt.ion()  # Turn on interactive mode
        # plt.get_current_fig_manager().full_screen_toggle()

    def small_plot(self, x, y1, y2):

        # Update the line data
        self.line1.set_xdata(x)
        self.line1.set_ydata(y1)
        self.line2.set_xdata(x)
        self.line2.set_ydata(y2)

        self.ax.relim()
        self.ax.autoscale_view()

        # Draw and pause for update
        plt.draw()
        plt.pause(0.001)

    def saveGraph(self, name):
        plt.savefig(str(name), dpi=300)

    def show(self):
        plt.ioff()  # Turn off interactive mode
        plt.show()