#include "BYTETracker.h"
#include <fstream>

BYTETracker::BYTETracker(double fw , double fh, GrowConfig growconfig, 
	float k1, float k2, float k3,
	float kl1, float kl2, float kl3,
	float focal_length_x, float focal_length_y,
	WeightInfer *weightInferPtr, int cam_height, float cam_center_x, float cam_center_y,
	int max_num_tracks)
	: i2w(k1, k2, k3, kl1, kl2, kl3, cam_height, cam_center_x, cam_center_y, focal_length_x, focal_length_y)
{
	// ByteTracker 추적 성능 관련 지표
	track_thresh = 0.5; // 이 수치 이상이면 detection_high 아니라면 detection_low
	high_thresh = track_thresh + 0.1; // 이 수치 이상이면 새로운 객체라 판단

	// NOTE: 값이 떨어질수록 더 엄격하게 보겠다는 의미? iou가 0.6/2 보다 낮아야 의미있음
	// 헝가리안 최적해(각 행 매칭의 합의 최솟값을 매칭기준으로 삼는데)는 lost값이 있는 경우니 매칭이 안되는 1개의 행이 존재한다
	// 그 행의 값은 match_threshold의 절반(대각대칭행렬)으로 아래의 예시에서는 0.3이 된다. 
	// 따라서 최적해를 찾을 떄, 객체가 1개 있는 경우 iou가 0.5일 때를 가정하면, 헝가리안 input 행렬의 값은 1-0.5 = 0.5가 되고
	// ((0.5, 0.3), (0.3, 0))의 input행렬이 만들어진다. 
	// 위에서 최적이 될 수 있는 공식은 0.5 + 0 < 0.3 + 0.3 보다 낮으니 성립이 된다.
	// match_threshold = 0.3 일 땐, ((0.5, 0.15), (0.15, 0)) 로 0.4 > 0.15 + 0.15 보다 크니, 객체를 추적하지 않는 것으로 결론짓는다.
	// 따라서 match_threshold가 높을수록 더 널널하게 트래킹한다는 의미
	match_thresh = 0.8; 
	_distance_thresh = 0.1; // ANCHOR : 알고리즘 변경으로 인해 항상 고정 0.8
	_weight_rbbox = 0.8;
	_weight_distance = 0.2;
	// max_time_lost = 10 * 86400; // fps * sec
	_max_time_lost_sec = 3600 * 6; // sec

	_max_num_tracks = max_num_tracks;

	frame_id = 0;

	// Edgefarm 기본 output
	_img_col = fw;
	_img_row = fh;

	// Edgefarm 복도 바닥판 실험 결과 필요
	mStandardPixel = growconfig.pixel;
	mStandardCm = growconfig.cm;

	// Chessboard 무게측정영역
	_chx1 = growconfig.chx1;
	_chy1 = growconfig.chy1;
	_chx2 = growconfig.chx2;
	_chy2 = growconfig.chy2;

	// Weight 보정치
	_weight_bias = growconfig.weight_bias;

	_roomid = growconfig.roomID;

	// 신규 식사영역 추가
	EatArea = growconfig.EatArea;
	_food_area_vec = growconfig.food_area_vec; // 제거 예정


	_spray_vec = growconfig.spray_vec;

	// weight infer
	_weightInferPtr = weightInferPtr;
	_cam_height = cam_height;

	float L0 = ((cam_height - 220.0f) * (cam_height - 230.0f)) / 200.0f;
	float L1 = ((cam_height - 210.0f) * (cam_height - 230.0f)) / -100.0f;
	float L2 = ((cam_height - 210.0f) * (cam_height - 220.0f)) / 200.0f;
	_cam_height_factor = L0 * (1.0f - 0.08f) + L1 * 1.0f + L2 * (1 + 0.08f);

	_cam_center_x = cam_center_x;
	_cam_center_y = cam_center_y;
	
}

BYTETracker::~BYTETracker()
{
}

vector<STrack> BYTETracker::update(const vector<BYTEObject> &objects, int src_id)
{

	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	vector<STrack> activated_stracks;
	vector<STrack> tentative_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> refind_stracks;
	vector<STrack> rematched_stracks;
	vector<STrack> removed_stracks;

	vector<STrack> detections;
	vector<STrack> detections_low;

	vector<STrack> detections_cp;
	vector<STrack> tracked_stracks_swap;
	vector<STrack> resa, resb;
	vector<STrack> output_stracks;

	vector<STrack *> unconfirmed;
	vector<STrack *> tracked_stracks;
	vector<STrack *> strack_pool;
	vector<STrack *> r_tracked_stracks;

	auto now = chrono::system_clock::now();

	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].rect.x;
			tlbr_[1] = objects[i].rect.y;
			tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
			tlbr_[3] = objects[i].rect.y + objects[i].rect.height;
			float rad = objects[i].rad;
			float landmarks_x1 = objects[i].landmarks_x1;
			float landmarks_y1 = objects[i].landmarks_y1;
			float landmarks_x2 = objects[i].landmarks_x2;
			float landmarks_y2 = objects[i].landmarks_y2;
			float landmarks_x3 = objects[i].landmarks_x3;
			float landmarks_y3 = objects[i].landmarks_y3;

			float landmarks_x4 = objects[i].landmarks_x4;
			float landmarks_y4 = objects[i].landmarks_y4;
			float landmarks_x5 = objects[i].landmarks_x5;
			float landmarks_y5 = objects[i].landmarks_y5;
			float landmarks_x6 = objects[i].landmarks_x6;
			float landmarks_y6 = objects[i].landmarks_y6;

			float landmarks_x7 = objects[i].landmarks_x7;
			float landmarks_y7 = objects[i].landmarks_y7;
			float landmarks_x8 = objects[i].landmarks_x8;
			float landmarks_y8 = objects[i].landmarks_y8;
			float landmarks_x9 = objects[i].landmarks_x9;
			float landmarks_y9 = objects[i].landmarks_y9;

			float score = objects[i].prob;
			int cls = objects[i].cls;

			float cx = objects[i].rect.x + (objects[i].rect.width / 2);
			float cy = objects[i].rect.y + (objects[i].rect.height / 2);

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, rad, cls,
				landmarks_x1, landmarks_y1, landmarks_x2, landmarks_y2, landmarks_x3, landmarks_y3, 
				landmarks_x4, landmarks_y4, landmarks_x5, landmarks_y5, landmarks_x6, landmarks_y6, 
				landmarks_x7, landmarks_y7, landmarks_x8, landmarks_y8, landmarks_x9, landmarks_y9,
				_img_col, _img_row, _weight_bias, mStandardPixel, mStandardCm, 
				_chx1, _chy1, _chx2, _chy2, _spray_vec, _roomid, src_id,
				&i2w);

			if (score >= track_thresh) detections.push_back(strack);
			else detections_low.push_back(strack);
			
		}
	}

	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		// if (!this->tracked_stracks[i].is_activated){
		// 	unconfirmed.push_back(&this->tracked_stracks[i]);
		// }
		// else
		// 	tracked_stracks.push_back(&this->tracked_stracks[i]);

		tracked_stracks.push_back(&this->tracked_stracks[i]);
	}

	for (int i = 0; i < this->tentative_stracks.size(); i++)
	{
		unconfirmed.push_back(&this->tentative_stracks[i]);
	}

	////////////////// Step 2: First association, with IoU //////////////////
	STrack::multi_predict(tracked_stracks, this->kalman_filter);
	STrack::multi_predict(unconfirmed, this->kalman_filter);
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	// STrack::multi_predict(strack_pool, this->kalman_filter);

	vector<vector<float>> dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	vector<vector<int>> matches;
	vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);
	
	for (int i = 0; i < matches.size(); i++)
	{
		// cout << "T : " << matches[i][0] << " D : " << matches[i][1] << " | Matching True" << endl;
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id, _TrackInformation, EatArea, 
				now,
				_weightInferPtr, _cam_height, _cam_center_x, _cam_center_y);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
			// std::cout << "\n[Ch " << src_id << "] Refind >> " << track->track_id << std::endl;
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());

	for (int i = 0; i < u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id, _TrackInformation, EatArea,
				now,
				_weightInferPtr, _cam_height, _cam_center_x, _cam_center_y);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
			// std::cout << "\n[Ch " << src_id << "] Refind >> " << track->track_id << std::endl;
		}
	}

	for (int i = 0; i < u_track.size(); i++)
	{
		STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
			// std::cout << "\n[Ch " << src_id << "] Lost >> " << track->track_id << std::endl;
		}
	}

	// cout << "[STEP3-2]" << endl;
	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	vector<STrack *> to_activate_tracks;
	for (int i = 0; i < matches.size(); i++)
	{
		// unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id, _TrackInformation, EatArea,
		// 	_weightInferPtr, _cam_height, _cam_center_x, _cam_center_y);

		// activated_stracks.push_back(*unconfirmed[matches[i][0]]);

		STrack *track = unconfirmed[matches[i][0]];
		STrack *det = &detections[matches[i][1]];

		// 일정 프레임 이상 계속 매칭되면 활성화 모드로 전환.
		if (this->frame_id - track->start_frame > _probation_age){
			track->update(*det, this->frame_id, _TrackInformation, EatArea,
				now,
				_weightInferPtr, _cam_height, _cam_center_x, _cam_center_y);

			to_activate_tracks.push_back(track);
		}else{
			track->update(*det, this->frame_id, _TrackInformation, EatArea,
				now,
				_weightInferPtr, _cam_height, _cam_center_x, _cam_center_y);

			tentative_stracks.push_back(*track);
		}
	}

	// Lost 애들과 리매칭.
	vector<STrack *> already_lost_stracks;
	for (int i = 0; i < this->lost_stracks.size(); i++){
		STrack *track = &(this->lost_stracks[i]);
		if (track->state == TrackState::Lost){
			already_lost_stracks.push_back(track);
		}
	}
	int num_of_already_lost_stracks = already_lost_stracks.size();

	if (to_activate_tracks.size() > 0){
		if (num_of_already_lost_stracks > 0){
			dists = diou_distance(already_lost_stracks, to_activate_tracks, dist_size, dist_size_size);
			matches.clear();
			vector<int> u_already_lost_stracks, u_to_activate_tracks;
			linear_assignment(dists, dist_size, dist_size_size, 999999.0, matches, u_already_lost_stracks, u_to_activate_tracks);
		
			// NOTE 리매칭. 매칭된 애들
			for (int i = 0; i < matches.size(); i++)
			{
				STrack *track = already_lost_stracks[matches[i][0]];
				STrack *det = to_activate_tracks[matches[i][1]];
				// 리매칭에 쓰인 활성화 추적 객체는 바로 소멸.
				track->re_match(*det, this->frame_id);
				det->state = TrackState::Removed;
				rematched_stracks.push_back(*track);
				// std::cout << "\n[Ch " << src_id << "] Rematched >> " << track->track_id << std::endl;

				// _TrackInformation에서 지우기.
				auto it = _TrackInformation.find(det->track_id);
				if (it != _TrackInformation.end()) {
					_TrackInformation.erase(it);
				}
			}

			// 매칭 안된 lost들
			// u_already_lost_stracks

			// 매칭할 Lost가 없는 activate 된 애들.
			for (int i = 0; i < u_to_activate_tracks.size(); i++){
				STrack *track = to_activate_tracks[u_to_activate_tracks[i]];
				int tentative_track_id = track->track_id;
				track->activate();
				activated_stracks.push_back(*track);
				// std::cout << "\n[Ch " << src_id << "] Activate >> " << track->track_id << std::endl;

				// _TrackInformation에서 지우기.
				auto it = _TrackInformation.find(tentative_track_id);
				if (it != _TrackInformation.end()) {
					_TrackInformation.erase(it);
				}
			}

		}else{
			for (int i = 0; i < to_activate_tracks.size(); i++){
				STrack *track = to_activate_tracks[i];
				int tentative_track_id = track->track_id;
				track->activate();
				activated_stracks.push_back(*track);
				// std::cout << "\n[Ch " << src_id << "] Activate >> " << track->track_id << std::endl;

				// _TrackInformation에서 지우기.
				auto it = _TrackInformation.find(tentative_track_id);
				if (it != _TrackInformation.end()) {
					_TrackInformation.erase(it);
				}
			}
		}
	}


	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack *track = unconfirmed[u_unconfirmed[i]];
		if (this->frame_id - track->end_frame() > _early_termination_age){ // 수습기간일 때는 더 세게 처리
			track->mark_removed();

			// _TrackInformation에서 지우기.
			auto it = _TrackInformation.find(track->track_id);
			if (it != _TrackInformation.end()) {
				_TrackInformation.erase(it);
			}

			// removed_stracks.push_back(*track);
			// std::cout << "\nRemove (Tentative)Ch [" << src_id << "]  >> " << track->track_id << std::endl;
		}else{
			tentative_stracks.push_back(*track);
		}
	}

	// * Step 4: Init new stracks
	for (int i = 0; i < u_detection.size(); i++)
	{
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->high_thresh)
			continue;

		// 최대 추적 수 제한.
		if (_max_num_tracks > 0){
			if (_max_num_tracks <= STrack::_count_vec[src_id] - num_of_already_lost_stracks){
				continue;
			}
		}
			
		// 임시 활성화 모드로 진입.
		track->tentative(this->kalman_filter, this->frame_id);

		tentative_stracks.push_back(*track);
	}

	// * Step 5: Update state
	for (int i = 0; i < this->lost_stracks.size(); i++)
	{
		STrack *track = &(this->lost_stracks[i]);
		// if (this->frame_id - track->end_frame() > this->max_time_lost)
		if (now - track->_last_updated > std::chrono::seconds(_max_time_lost_sec)) {
			track->mark_removed();
			removed_stracks.push_back(*track);
		}
	}
	

	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, rematched_stracks);

	// std::cout << activated_stracks.size() << std::endl;

	vector<STrack> tentative_stracks_swap;

	for (int i = 0; i < this->tentative_stracks.size(); i++)
	{
		if (this->tentative_stracks[i].state == TrackState::Tentative)
		{
			tentative_stracks_swap.push_back(this->tentative_stracks[i]);
		}
	}
	this->tentative_stracks.clear();
	this->tentative_stracks.assign(tentative_stracks_swap.begin(), tentative_stracks_swap.end());

	this->tentative_stracks = joint_stracks(this->tentative_stracks, tentative_stracks);

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	this->removed_stracks = joint_stracks(this->removed_stracks, removed_stracks);

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);

	// for (int i = 0; i < removed_stracks.size(); i++)
	// {
	// 	this->removed_stracks.push_back(removed_stracks[i]);
	// }

	// 벡터를 청소해줘야 함 (그렇지 않으면 무수히 쌓임 - 메모리 누수)
	if (this->removed_stracks.size() > 100) this->removed_stracks.clear();

	// // 중복제거 (Lost 추적 객체가 제거될 위험이 있으므로 사용하지 않음.)
	// remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);
	// this->tracked_stracks.clear();
	// this->tracked_stracks.assign(resa.begin(), resa.end());
	// this->lost_stracks.clear();
	// this->lost_stracks.assign(resb.begin(), resb.end());

	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}

		// output_stracks.push_back(this->tracked_stracks[i]);
	}

	// 최대 마리 수 기록
	occupancy = max(occupancy, static_cast<int>(output_stracks.size()));


	// // NOTE DEBUG: _TrackInformation 체크
	// if (now - last_debug_check_time >= std::chrono::seconds(3)){
	// 	int capacity = 0;
	// 	std::ostringstream track_id_list_oss;

	// 	for (auto& pair : _TrackInformation) {
	// 	    int trackID = pair.first;
	// 	    TrackInfo& info = pair.second;
	// 		capacity += info.input_vector_for_dnn.capacity();
	// 		track_id_list_oss << trackID;
	// 		track_id_list_oss << ", ";
	// 	}

	// 	std::string track_id_list_str = track_id_list_oss.str();

	// 	std::cout << "CAM " << src_id << "'s TrackInformation: len=" << _TrackInformation.size() << ", track id list = " << track_id_list_str << std::endl;

	// 	last_debug_check_time = now;
	// }

	std::vector<bool> feedDetected(EatArea.size(), false);
	isFeedAreaEmpty.clear();
	isFeedAreaEmpty.reserve(EatArea.size());

	// 각 트랙 객체에서 급이 영역에 포함되었는지 검사
	for (auto &track : output_stracks) {
		for (size_t i = 0; i < EatArea.size(); i++) {
			if (track.CheckEatingForZone(EatArea[i])) {
				feedDetected[i] = true;
			}
		}
	}

	// feedDetected[i]가 false면 먹이 영역이 비어있다는 뜻이므로
	for (bool detected : feedDetected) {
		isFeedAreaEmpty.push_back(!detected);
	}

	return output_stracks;
}

vector<STrack> BYTETracker::get_lost_stracks()
{
	return this->lost_stracks;
}

vector<STrack> BYTETracker::get_tentative_stracks()
{
	return this->tentative_stracks;
}

vector<STrack> BYTETracker::interval()
{
	vector<STrack> output_stracks;
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}
	return output_stracks;
}

void BYTETracker::FullOverlayImg(cv::Mat &img) {
	OverlayWeightArea(img);
	OverlaySprayShootingArea(img);
}

void BYTETracker::OverlayWeightArea(cv::Mat &img) {
	cv::rectangle(img, cv::Rect2f(_chx1, _chy1, _chx2 - _chx1, _chy2 - _chy1), cv::Scalar(0,0,255), 2, 8, 0);
	// cv::rectangle(img, cv::Rect(img.cols * 0.1, img.rows * 0.1, img.cols * 0.8, img.rows * 0.8), cv::Scalar(0,255,0), 2, 8, 0);
}

void BYTETracker::OverlaySprayShootingArea(cv::Mat &img) {
	for (int i = 0; i < _spray_vec.size(); i++) {
		cv::circle(img, cv::Point2f(_spray_vec[i].set_cx, _spray_vec[i].set_cy), _spray_vec[i].set_r, cv::Scalar(255, 0, 0), 1, 8, 0);
		cv::putText(img, to_string(_spray_vec[i].spray_id), cv::Point2f(_spray_vec[i].set_cx-6, _spray_vec[i].set_cy+6), 1, 1.5, cv::Scalar(255, 0, 0), 2);
	}
}

void BYTETracker::Reset_Meta(bool erase_option) {

	// if (this->tracked_stracks.size() > 0) this->tracked_stracks[0].reset_id();
	// if (this->lost_stracks.size() > 0) this->lost_stracks[0].reset_id();
	// if (this->removed_stracks.size() > 0) this->removed_stracks[0].reset_id();

	// this->tracked_stracks.clear();
	// this->lost_stracks.clear();
	this->removed_stracks.clear();

	// _TrackInformation.clear();
	for (auto& pair : _TrackInformation) {
        int trackID = pair.first;
        TrackInfo& info = pair.second;

		// 10분마다 측정해야하는 것 reset
		info.total_move_cm = 0;
		info.feedFrame = 0;
		info.input_vector_for_dnn_byframe.clear();

		// Weight 관련 초기화 (10분마다 초기화)
		info.frame_weight_cnt = 0;
		info.med = 0.0f;
		info.med_bodycm = 0.0f;
		info.med_shouldercm = 0.0f;
	}

	if (erase_option) {
		auto now = chrono::system_clock::now();
		for (auto it = _TrackInformation.begin(); it != _TrackInformation.end(); ) {
			int trackID = it->first;
			TrackInfo& info = it->second;

			// 수습 기간(임시활성화)된 애들의 정보는 무시.
			if (trackID < 0){
				++it;
				continue;
			}					

			auto endTime = ConvertStringToTimePoint(info.eTime);
			auto timeDifference = std::chrono::duration_cast<std::chrono::seconds>(now - endTime).count();

			if ((info.tracklet_len >= 100 && timeDifference <= TimeCycle && timeDifference > 0) || 
				(info.tracklet_len < 100 && timeDifference <= std::max(int(TimeCycle/6), 100) && timeDifference > 0)) {
				// 조건을 만족하면 대기
				++it;
			}
			else {
				// // 조건을 만족하지 않으면 해당 트랙을 제거
				// it = _TrackInformation.erase(it); // 요소 삭제 후 반복자 업데이트

				// 반복자만 이동
				++it;
			}

			// Lost 시간 초과 된 애들 삭제
			if (std::chrono::seconds(timeDifference) > std::chrono::seconds(_max_time_lost_sec)){
				it = _TrackInformation.erase(it); // 요소 삭제 후 반복자 업데이트
			}
		}
	}

	occupancy = 0;

}


string BYTETracker::MakeHourLog(const string& serial_number, const int& camID, std::chrono::time_point<std::chrono::system_clock> created_log_time, const double& fps, const int& age, const string& standard_data, const int& target_age, string outfolder) {

	if (outfolder == "") outfolder = "/edgefarm_config/Recording/";

	// NameMaker
	time_t now_time = chrono::system_clock::to_time_t(created_log_time);
	tm* local_time = localtime(&now_time);

	int fps_int = static_cast<int>(fps);
	int fps_decimal = static_cast<int>(round((fps - fps_int) * 10));
	stringstream fps_ss;
	if (fps_decimal == 10) {
		fps_int += 1;
		fps_decimal = 0;
	}
	fps_ss << fps_int << "o" << fps_decimal;

	stringstream new_filename_ss;
	new_filename_ss << "efg_report_" << serial_number << "_"
					<< camID << "_"
					<< put_time(local_time, "%y%m%d_%H%M%S") << "_"
					<< occupancy << "_"
					<< age << "_"
					<< standard_data << "_"
					<< target_age << "_"
					<< fps_ss.str() << ".txt";

	string savepath = outfolder + new_filename_ss.str();
	
	// 파일 열기
    ofstream outfile(savepath, ios::app); // 'ios::app' is to append to the file

	if (!outfile.is_open()) {
        cout << "Failed to open the file: " << savepath << endl;
		return "";
    }

	// 반복자를 사용하여 안전하게 요소를 삭제하기 위해 수정된 루프
    for (auto it = _TrackInformation.begin(); it != _TrackInformation.end(); ) {
        int trackID = it->first;
        TrackInfo& info = it->second;

		// 수습 기간(임시활성화)된 애들의 정보는 무시.
		if (trackID < 0){
			++it;
			continue;
		}		

		auto endTime = ConvertStringToTimePoint(info.eTime);
		auto timeDifference = std::chrono::duration_cast<std::chrono::seconds>(created_log_time - endTime).count();

		// NOTE: DNN weight로 med값 변경
		if (_weightInferPtr != nullptr) {
			info.med = sampling_dnn_weight(info.input_vector_for_dnn_byframe, info.std);
			cout << "[DNN] info.med: " << info.med << endl;
			// med가 0이 되는 case는 trackID의 frame수가 N=10 보다 작아서 샘플링 실패했을 경우
			if (info.med > 0) info.med += _weight_bias;
		}


		// 100f 이상이면서 1시간 이내에 갱신된 trackid 
		// 100f 이하여도 3분 이내에 갱신된 trackid
		// 혹시나 하루가 지났을 경우 대비
        if ((info.tracklet_len >= 100 && timeDifference <= TimeCycle && timeDifference >= 0) || (info.tracklet_len < 100 && timeDifference <= std::max(int(TimeCycle/6), 100) && timeDifference >= 0)) {
            // 파일에 트랙 정보 기록
            outfile << trackID << ",";
            outfile << info.frame_weight_cnt << ",";
            outfile << to_string_with_precision(info.med, 2) << ",";
            outfile << to_string_with_precision(info.med_bodycm, 2) << ",";
            outfile << to_string_with_precision(info.med_shouldercm, 2) << ",";
            outfile << info.sTime << ",";
            outfile << info.eTime << ",";
            outfile << info.feedFrame << ",";
            outfile << to_string_with_precision(info.total_move_cm, 2) << "\n";

			// checked = true;

            // 반복자만 이동
            ++it;
        } 
		else {
            // // 조건을 만족하지 않으면 해당 트랙을 제거
            // it = _TrackInformation.erase(it); // 요소 삭제 후 반복자 업데이트

            // 반복자만 이동
            ++it;
        }

		// Lost 시간 초과 된 애들 삭제
		if (std::chrono::seconds(timeDifference) > std::chrono::seconds(_max_time_lost_sec)){
	    	it = _TrackInformation.erase(it); // 요소 삭제 후 반복자 업데이트
		}

		// std::cout << trackID << ": " << checked << ", " << ", " << info.tracklet_len << ", " << info.eTime << ", " << timeDifference  << std::endl;
    }

    outfile.close(); // 파일 닫기

	Reset_Meta();
	return new_filename_ss.str(); // 파일명만 
}

// 시간 문자열을 time_t로 변환하는 함수 (HHMMSS 형식)
std::chrono::system_clock::time_point BYTETracker::ConvertStringToTimePoint(const std::string& timeString) {
    // 현재 날짜를 기준으로 HHMMSS 시간 변환
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_time_t);

    // 시간 문자열을 파싱
    std::istringstream ss(timeString);
    ss >> std::get_time(&tm, "%H%M%S");

    // 변환한 tm을 시간점으로 변환
    return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

float BYTETracker::sampling_dnn_weight(const vector<vector<float>> &input_vector_for_dnn_byframe, float &stddev_weight) {
	int n_repeat = 30;

	// ANCHOR 샘플링 후 무게 계산
	int N = input_vector_for_dnn_byframe.size();

	if (N < _weightInferPtr->sequance) return 0.0f;

	vector<float> dnn_weight_vec;

	for (int n=0; n<n_repeat; n++) {
		std::vector<std::vector<float>> sampled_frames;
		static std::mt19937 rng{ std::random_device{}() };
		sampled_frames.clear();
		std::sample(input_vector_for_dnn_byframe.begin(),
					input_vector_for_dnn_byframe.end(),
					std::back_inserter(sampled_frames),
					_weightInferPtr->sequance,
					rng);
					
		int frame_dim = sampled_frames[0].size(); 
		int k = sampled_frames.size();
		std::vector<float> dnn_input;
		dnn_input.reserve(frame_dim * k); // 공간 확보
		for (auto& f : sampled_frames) dnn_input.insert(dnn_input.end(), f.begin(), f.end());
		float dnn_weight = _weightInferPtr->run_inference(dnn_input) * _cam_height_factor;
		dnn_weight_vec.push_back(dnn_weight);
	}

	float mean_weight = std::accumulate(dnn_weight_vec.begin(), dnn_weight_vec.end(), 0.0f) / dnn_weight_vec.size();
	float sum_of_squares = std::accumulate(dnn_weight_vec.begin(), dnn_weight_vec.end(), 0.0f, 
		[mean_weight](float acc, float val) {
			return acc + (val - mean_weight) * (val - mean_weight);
		});
	stddev_weight = std::sqrt(sum_of_squares / dnn_weight_vec.size());

	return mean_weight;

}