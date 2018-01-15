# implement value iteration

# define T(s,a,s')
# first line = [T(A,0,A),T(A,0,B)]
# second line = [T(A,1,A),T(A,1,B)]
# thrid line = [T(B,0,A),T(B,0,B)]
# fourth line = [T(B,1,A),T(B,1,B)]

T = [[0.50,0.50],
	 [0.50,0.50],
	 [0.00,1.00],
	 [0.10,0.90]]

# define R(s,a,s')
# first line = [R(A,0,A),R(A,0,B)]
# second line = [R(A,1,A),R(A,1,B)]
# thrid line = [R(B,0,A),R(B,0,B)]
# fourth line = [R(B,1,A),R(B,1,B)]

R = [[2.0,-1.0],
	 [1.0,2.0],
	 [-2.0,-1.0],
	 [-3.0,-1.0]]

# define Q(s,a)
# first line = [Q(A,0),Q(A,1)]
# second line = [Q(B,0),Q(B,1)]

Q = [[0,0],
	 [0,0]]

def value_iteration(transition,reward):
	result_A, result_B = 0, 0
	for i in range(2):
		Q[0][0] = (T[0][0] * (R[0][0] + result_A)) + (T[0][1] * (R[0][1] + result_B))
		Q[0][1] = (T[1][0] * (R[1][0] + result_A)) + (T[1][1] * (R[1][1] + result_B))
		Q[1][0] = (T[2][0] * (R[2][0] + result_A)) + (T[2][1] * (R[2][1] + result_B))
		Q[1][1] = (T[3][0] * (R[3][0] + result_A)) + (T[3][1] * (R[3][1] + result_B))
		print "================="
		print result_A
		print result_B
		print Q[0][0], Q[0][1]
		print Q[1][0], Q[1][1]
		print "================="
		result_A = max(Q[0][0],Q[0][1])
		result_B = max(Q[1][0],Q[1][1])

	print "End with : "
	print result_A
	print result_B


def main():
	value_iteration(T,R)

if __name__ == '__main__':
	main()