# implement value iteration

from Tkinter import *

#start & end
point_start = [0,4]
point_end = [4,2]

# size
size = (5,7)

# define gamma
gamma = 0.9

# define T(s,a,s')
# North, East, South, West
T = [[0.9,0.1],
	 [0.9,0.1],
	 [0.9,0.1],
	 [0.9,0.1]]

# define s'
# North, East, South, West
S_future = [[(0,1),(-1,0)],
			[(1,0),(0,-1)],
			[(0,1),(-1,0)],
			[(-1,0),(0,1)]]

# define R(s,a,s')
# -100 = a wall
R = [[0 for i in range (7)] for j in range (5)]
R[4][2] = 100
R[4][1], R[4][3] = -100, -100
# R[1][4] = -100

# define Q(s,a)
# North, East, South, West
Q = [[[0,0,0,0] for i in range (7)] for j in range (5)]

# best result
best = [[0 for i in range (7)] for j in range (5)]

def neighbour(position):
	x,y = position
	n = []

	if (x-1 >= 0):
		n.append([x-1,y])
	else:
		n.append([x,y])
	if (y-1 >= 0):
		n.append([x,y-1])
	else:
		n.append([x,y])
	if (x+1 < 5):
		n.append([x+1,y])
	else:
		n.append([x,y])
	if (y+1 < 7):
		n.append([x,y+1])
	else:
		n.append([x,y])

	return n

def value_iteration():
	state = neighbour(point_end)
	#while point_start not in state:
	for i in range (10):
		new_n = []
		for s in state:

			s_x,s_y = s
			x_north = s_x+1 if (s_x+1) < 5 else s_x
			x_south = s_x if (s_x-1) <= 0 else s_x-1
			y_east = s_y if (s_y-1) <= 0 else s_y-1
			y_west = s_y+1 if (s_y+1) < 7 else s_y
			Q[s_x][s_y][0] = (T[0][0]) * (R[x_north][s_y] + best[x_north][s_y]) + (T[0][1] * (R[s_x][y_west] + best[s_x][y_west]))
			Q[s_x][s_y][1] = (T[1][0]) * (R[s_x][y_east] + best[s_x][y_east]) + (T[1][1] * (R[x_south][s_y] + best[x_south][s_y]))
			Q[s_x][s_y][2] = (T[2][0]) * (R[x_south][s_y] + best[x_south][s_y]) + (T[2][1] * (R[s_x][y_east] + best[s_x][y_east]))
			Q[s_x][s_y][3] = (T[3][0]) * (R[s_x][y_west] + best[s_x][y_west]) + (T[3][1] * (R[x_north][s_y] + best[x_north][s_y]))

			new_n += neighbour(s)

			best[s_x][s_y] = max(Q[s_x][s_y])
			
		for n in new_n:
			if n not in state:
				state += [n]

	# print best

def best_way(start,end):
	point_list = [start]
	point = start
	while end not in point_list:
		n = neighbour(point)
		reward = [best[i][j] for (i,j) in n]
		best_reward = max(reward)
		best_point = n[reward.index(best_reward)]

		point_list += [best_point]
		point = best_point

	print point_list
	return point_list

def main():
	# print Q
	value_iteration()
	way = best_way(point_start,point_end)

if __name__ == '__main__':
	main()