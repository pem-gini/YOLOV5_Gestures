import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KalmanObject:
    def __init__(self, obj):
        ## set other kwargs from dict
        for key, value in obj.__dict__.items():
            setattr(self, key, value)

        self.kf = KalmanFilter(dim_x=3, dim_z=3)
        self.kf.F = np.array([[1, 0, 0 ],
                              [0, 1, 0 ],
                              [0, 0, 1 ]])
        self.kf.H = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        self.kf.P *= 1000.
        self.kf.R *= 5.
        self.kf.Q = Q_discrete_white_noise(dim=3, dt=1, var=0.1)
        self.kf.x = np.array([self.x, self.y, self.z])
        self.hitcycles = 0
        self.lifecycles = 1
        self.misscycles = 0
        self.updated = True

    def get(self):
        return self.kf.x.tolist()
    
    def predict(self):
        self.kf.predict()

    def tick(self):
        self.lifecycles = self.lifecycles + 1

    def update(self, obj):
        ## set other kwargs from dict
        for key, value in obj.__dict__.items():
            setattr(self, key, value)
        x,y,z = (obj.x, obj.y, obj.z)
        
        self.kf.update(np.array([x, y, z]).reshape(3, 1))
        self.updated = True
        self.misscycles = 0

    def calculateRatio(self):
        return self.hitcycles / self.lifecycles
    
    def __repr__(self):
        return f" - (prob={self.calculateRatio()} |-- {self.misscycles}, x={self.kf.x[0]}, y={self.kf.x[1]}, z={self.kf.x[2]})"


class KalmanGestureTracker:
    def __init__(self, distance_threshold=5.0):
        self.tracks = {}
        self.next_id = 1
        self.distance_threshold = distance_threshold

    def update(self, new, f=None):
        ### tick all hypothesis once
        for track_id, obj in self.tracks.items():
            obj.tick()
            obj.predict()
        ### process all incoming points
        for n in new:
            self.add_or_update_obj(n, f)
        ### cleanup dead object hypothesis
        cleanups = []
        for track_id, obj in self.tracks.items():
            ### if object got updated this cycle, add hitcount
            if obj.updated:
                obj.hitcycles += 1
            else:
                obj.misscycles += 1
            ratio = obj.calculateRatio()
            # print(obj.hitcycles, obj.lifecycles, obj.misscycles)
            ### clear upated flag 
            obj.updated = False
            ### delete if necessary
            if obj.misscycles > 20 or obj.lifecycles > 5 and ratio < 0.3:
                cleanups.append(track_id)
        for key in cleanups:
            # print("##########################################################################")
            # print("########################## %s #########################################" % key)
            # print("##########################################################################")
            del self.tracks[key]

    def add_or_update_obj(self, newobj, f=None):
        new_point = np.array([newobj.x, newobj.y, newobj.z])
        closest_track_id = None
        min_distance = float('inf')
        for track_id, obj in self.tracks.items():
            track_position = obj.get()
            conditionals = True
            if f:
                conditionals = f(obj, newobj)
            dist = np.linalg.norm(track_position - new_point)
            if conditionals and dist < min_distance:
                min_distance = dist
                closest_track_id = track_id
        if closest_track_id is not None and min_distance <= self.distance_threshold:
            self.tracks[closest_track_id].update(newobj)
        else:
            self.tracks[self.next_id] = KalmanObject(newobj)
            self.next_id += 1

    def get_all_objects(self):
        return {track_id: obj for track_id, obj in self.tracks.items()}

    def __repr__(self):
        return f"KalmanPointTracker(tracks={self.tracks})"

    def plot_tracks(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for track_id, obj in self.tracks.items():
            p = obj.get()
            x,y,z = (p[0], p[1], p[2])
            ax.plot(x,y,z, label=f'Track {track_id}')
            ax.text(int(x),int(y),int(z), f"{obj.cls}", size=10, zorder=1, color='k')
            ax.scatter(x,y,z, s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        return ax


### Example usage
if __name__ == "__main__":
    class MyObject:
        def __init__(self, cls, x,y,z) -> None:
            self.cls = cls
            self.x = x
            self.y = y
            self.z = z
    ## create tracker
    tracker = KalmanGestureTracker(distance_threshold=8.0)
    ### create extra association function if necessary
    extraAssociationFunc = lambda old, new: old.cls == new.cls
    ### Adding or updating points
    objects = [
        MyObject(0, 1, 2, 3),
        MyObject(1, 2, 3, 1),
        MyObject(1, 3, 2.5, 2),
        MyObject(0, 4, 5, 6),
        MyObject(0, 1.5, 2.5, 3.5),
        MyObject(0, 10, 10, 10),
        MyObject(0, 12, 11, 9),
        MyObject(2, 11.5, 11, 9.5),
    ]
    tracker.update(objects, f=extraAssociationFunc)
    ### results
    print("1 Current positions:")
    for id, obj in tracker.get_all_objects().items():
        print(obj)
    ### Plotting the tracks
    ax = tracker.plot_tracks()
    ### some more updates for lifecycle check
    tracker.update([])
    tracker.update([])
    print("2 Current positions:")
    for id, obj in tracker.get_all_objects().items():
        print(obj)
    tracker.update([])
    tracker.update([])
    print("3 Current positions:")
    for id, obj in tracker.get_all_objects().items():
        print(obj)
    ### empty updates lead to degradation, next update should remove?
    tracker.update([])
    print("4 Current positions:")
    for id, obj in tracker.get_all_objects().items():
        print(obj)
    plt.show()
