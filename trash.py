def draw_points(surf, points, top=(0,0), color=C_RED):
	for point in points:
		pygame.draw.circle(surf, color, (point.astype(np.uint32) + top), 3)

def get_bound_box(kps_coord, kps_score):
	min_x = -1
	max_x = -1
	min_y = -1	
	max_y = -1
	
	for i in range(C_NKP):
		if kps_score[i] >= C_KP_THRESHOLD:
			x = kps_coord[i][0]
			y = kps_coord[i][1]
			
			if max_x == -1 or x > max_x:
				max_x = x
			elif min_x == -1 or x < min_x:
				min_x = x
			if max_y == -1 or y > max_y:
				max_y = y
			elif min_y == -1 or y < min_y:
				min_y = y
	
	return (min_x, min_y), (max_x, max_y)

def draw_pose(surf, kps_coord, kps_score, color=C_RED):
	for i in range(18):
		if kps_score[i] >= C_KP_THRESHOLD:
			pygame.draw.circle(surf, color, kps_coord[i], 3)
			
def draw_pose2(surf, kps_coord, kps_score, color=C_RED):
	if kps_score[C_LEYE] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_LEYE], 3)
	if kps_score[C_REYE] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_REYE], 3)
	if kps_score[C_LEAR] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_LEAR], 3)
	if kps_score[C_REAR] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_REAR], 3)
	if kps_score[C_NOSE] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_NOSE], 3)
		
	for pair in C_PAIRS:
		if kps_score[pair[0]] >= C_KP_THRESHOLD and kps_score[pair[1]] >= C_KP_THRESHOLD:
			pygame.draw.line(surf, color, kps_coord[pair[0]], kps_coord[pair[1]], 3)

def draw_bound_box(surf, kps_coord, kps_score, color=C_GREEN):
	top, bottom = get_bound_box(kps_coord, kps_score)
	pygame.draw.circle(surf, C_RED, top, 3)
	pygame.draw.circle(surf, C_RED, bottom, 3)
	pygame.draw.rect(surf, color, (top[0], top[1], bottom[0]-top[0], bottom[1]-top[1]), 3)
	return top, bottom
