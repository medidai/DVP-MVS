#include "main.h"
#include "APD.h"

using namespace boost::filesystem;

void setBit_YZL(unsigned int* input, const unsigned int n)
{
	(*input) |= (unsigned int)(1 << n);
}

// 求连通区域
void Connect_RGB(const cv::Mat& dst_image, cv::Mat& label_mask, std::vector<int>& label_cnt) {
	std::vector<std::vector<int>> left_neigh(dst_image.rows);
	std::vector<std::vector<int>> up_neigh(dst_image.rows);

	for (int y = 0; y < dst_image.rows; y++) {
		left_neigh[y].resize(dst_image.cols);
		up_neigh[y].resize(dst_image.cols);
		for (int x = 0; x < dst_image.cols; x++) {
			// 左连通
			if (x == 0) {
				left_neigh[y][x] = 0;
			}
			else {
				if (dst_image.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0) && dst_image.at<cv::Vec3b>(y, x - 1) == cv::Vec3b(0, 0, 0)) {
					left_neigh[y][x] = 1;
				}
				else {
					left_neigh[y][x] = 0;
				}
			}
			// 上连通
			if (y == 0) {
				up_neigh[y][x] = 0;
			}
			else {
				if (dst_image.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0) && dst_image.at<cv::Vec3b>(y - 1, x) == cv::Vec3b(0, 0, 0)) {
					up_neigh[y][x] = 1;
				}
				else {
					up_neigh[y][x] = 0;
				}
			}
		}
	}

	// 维护一个并查集
	int cnt = 1;
	std::vector<int> connection;
	connection.push_back(0);
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			if (dst_image.at<uchar>(y, x) == 255) {
				label_mask.at<int>(y, x) = 0;
			}
			else {
				bool left = false, up = false;
				if (left_neigh[y][x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					left = true;
				}
				if (up_neigh[y][x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y - 1, x);
					up = true;
				}
				if (left == false && up == false) {
					label_mask.at<int>(y, x) = cnt;
					connection.push_back(cnt);
					cnt++;
				}
				else if (left == true && up == true) {
					int left_label = label_mask.at<int>(y, x - 1);
					int up_label = label_mask.at<int>(y - 1, x);
					if (left_label > up_label) {
						connection[left_label] = up_label;
						label_mask.at<int>(y, x) = label_mask.at<int>(y - 1, x);
					}
					else if (left_label < up_label) {
						connection[up_label] = left_label;
						label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					}
				}
			}
		}
	}

	for (size_t i = 1; i < connection.size(); i++) {
		int cur_label = connection[i];
		int pre_label = connection[cur_label];
		while (pre_label != cur_label) {
			cur_label = pre_label;
			pre_label = connection[pre_label];
		}
		connection[i] = cur_label;
	}

	int label_num = 1;
	std::vector<int> mapping;
	mapping.push_back(0);
	for (size_t i = 1; i < connection.size(); i++) {
		mapping.push_back(0);
		if (connection[i] == (int)i) {
			mapping[i] = label_num;
			label_num++;  //标签总数
		}
	}

	// 重编号
	for (size_t i = 1; i < connection.size(); i++) {
		connection[i] = mapping[connection[i]];
	}

	for (int i = 0; i < label_num; i++) {
		label_cnt.push_back(0);
	}

	// 连通区域计数
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			int label = label_mask.at<int>(y, x);
			label_mask.at<int>(y, x) = connection[label];
			label_cnt[connection[label]]++;
		}
	}
}

void GenerateSampleList(const path& dense_folder, std::vector<Problem>& problems)
{
	path cluster_list_path = dense_folder / path("pair.txt");
	problems.clear();
	ifstream file(cluster_list_path);
	std::stringstream iss;
	std::string line;

	int num_images;
	iss.clear();
	std::getline(file, line);
	iss.str(line);
	iss >> num_images;

	for (int i = 0; i < num_images; ++i) {
		Problem problem;
		problem.index = i;
		problem.src_image_ids.clear();
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> problem.ref_image_id;

		problem.dense_folder = dense_folder;
		problem.result_folder = dense_folder / path("APD") / path(ToFormatIndex(problem.ref_image_id));
		create_directory(problem.result_folder);

		int num_src_images;
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> num_src_images;
		for (int j = 0; j < num_src_images; ++j) {
			int id;
			float score;
			iss >> id >> score;
			if (score <= 0.0f) {
				continue;
			}
			problem.src_image_ids.push_back(id);
		}
		problems.push_back(problem);
	}
}

bool CheckImages(const std::vector<Problem>& problems) {
	if (problems.size() == 0) {
		return false;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return false;
	}
	const int width = image.cols;
	const int height = image.rows;
	for (size_t i = 1; i < problems.size(); ++i) {
		image_path = problems[i].dense_folder / path("images") / path(ToFormatIndex(problems[i].ref_image_id) + ".jpg");
		image = cv::imread(image_path.string());
		if (image.cols != width || image.rows != height) {
			return false;
		}
	}
	return true;
}

void GetProblemEdges(const Problem& problem) {
	std::cout << "Getting image edges: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	int scale = 0;
	while ((1 << scale) < problem.scale_size) scale++;

	path image_folder = problem.dense_folder / path("images");
	path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
	cv::Mat image_uint = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
	cv::Mat src_img;
	image_uint.convertTo(src_img, CV_32FC1);
	const float factor = 1.0f / (float)(problem.scale_size);
	const int new_cols = std::round(src_img.cols * factor);
	const int new_rows = std::round(src_img.rows * factor);
	cv::Mat scaled_image_float;
	cv::resize(src_img, scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
	scaled_image_float.convertTo(src_img, CV_8UC1);

	if (problem.params.use_edge) {
		// path edge_path = problem.result_folder / path("edges.dmb");
		path edge_path = problem.result_folder / path("edges_" + std::to_string(scale) + ".dmb");
		std::ifstream edge_file(edge_path.string());
		bool edge_exists = edge_file.good();
		edge_file.close();
		if (!edge_exists) {
			cv::Mat edge = EdgeSegment(scale, src_img, 0, true);
			WriteBinMat(edge_path, edge);
			if (problem.show_medium_result) {
				path ref_image_edge_path = problem.result_folder / path("rawedge_" + std::to_string(scale) + ".jpg");
				cv::imwrite(ref_image_edge_path.string(), edge);
			}
		}
	}

	if (problem.params.use_label) {
		// path label_path = problem.result_folder / path("labels.dmb");
		path label_path = problem.result_folder / path("labels_" + std::to_string(scale) + ".dmb");
		std::ifstream label_file(label_path.string());
		bool label_exists = label_file.good();
		label_file.close();
		if (!label_exists) {
			cv::Mat label = EdgeSegment(scale, image_uint, 1);
			WriteBinMat(label_path, label);
			if (problem.show_medium_result) {
				path ref_image_con_path = problem.result_folder / path("connect_" + std::to_string(scale) + ".jpg");
				cv::imwrite(ref_image_con_path.string(), EdgeSegment(scale, image_uint, -1));
			}
		}
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Getting image edges: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
	std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

int ComputeRoundNum(const std::vector<Problem>& problems) {
	if (problems.size() == 0) {
		return 0;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return 0;
	}
	int max_size = MAX(image.cols, image.rows);
	int round_num = 1;
	while (max_size > 800) {
		max_size /= 2;
		round_num++;
	}
	return round_num;
}


void ProcessProblem(const Problem& problem) {
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	std::cout << "Iteration: " << problem.iteration << std::endl;

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	APD APD(problem);
	float depth_min = APD.GetDepthMin();
	float depth_max = APD.GetDepthMax();
	APD.InuputInitialization();
	APD.SupportInitialization();
	APD.CudaSpaceInitialization();
	APD.SetDataPassHelperInCuda();
	APD.RunPatchMatch();

	int width = APD.GetWidth(), height = APD.GetHeight();
	cv::Mat depth = cv::Mat(height, width, CV_32FC1);
	cv::Mat normal = cv::Mat(height, width, CV_32FC3);
	cv::Mat pixel_states = APD.GetPixelStates();

	// Visibility prior: per source view, pixels that did NOT select the view are
	// grouped into connected components and small ones are flipped to
	// "selected" (hole filling). Medida: rewritten on uchar masks and
	// parallelised over source views; upstream did this single-threaded on
	// CV_8UC3 mats and dominated runtime at fine scales.
	const int num_src = (int)problem.src_image_ids.size();
	std::vector<cv::Mat> selected_masks(num_src);
	for (int i = 0; i < num_src; ++i) {
		selected_masks[i] = cv::Mat(height, width, CV_8UC1);
	}

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			float4 plane_hypothesis = APD.GetPlaneHypothesis(r, c);
			depth.at<float>(r, c) = plane_hypothesis.w;
			if (depth.at<float>(r, c) < APD.GetDepthMin() || depth.at<float>(r, c) > APD.GetDepthMax()) {
				depth.at<float>(r, c) = 0;
				pixel_states.at<uchar>(r, c) = UNKNOWN;
			}
			normal.at<cv::Vec3f>(r, c) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);

			const unsigned int views = APD.GetPixelSelectedViews(r, c);
			for (int i = 0; i < num_src; ++i) {
				selected_masks[i].at<uchar>(r, c) = ((views >> i) & 1) ? 255 : 0;
			}
		}
	}

	const int min_hole_size = 20 * (8 / problem.scale_size) * (8 / problem.scale_size);
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_src; ++i) {
		cv::Mat lab_mask(height, width, CV_32S);
		std::vector<int> label_cnt;
		Connect(selected_masks[i], lab_mask, label_cnt);
		Label_Update(lab_mask, label_cnt);

		// label 0 = already-selected pixels; other labels = unselected components.
		std::vector<uchar> keep_unselected(label_cnt.size(), 0);
		for (size_t j = 1; j < label_cnt.size(); j++) {
			keep_unselected[j] = label_cnt[j] >= min_hole_size;
		}
		for (int y = 0; y < height; y++) {
			const int* labels = lab_mask.ptr<int>(y);
			uchar* mask = selected_masks[i].ptr<uchar>(y);
			for (int x = 0; x < width; x++) {
				mask[x] = keep_unselected[labels[x]] ? 0 : 255;
			}
		}
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned int temp_selected_views = 0;
			for (int i = 0; i < num_src; ++i) {
				if (selected_masks[i].at<uchar>(y, x) == 255)
					setBit_YZL(&temp_selected_views, i);
			}
			APD.SetPixelSelectedViews(y, x, temp_selected_views);
		}
	}

	path depth_path = problem.result_folder / path("depths.dmb");
	WriteBinMat(depth_path, depth);
	path normal_path = problem.result_folder / path("APD_normals.dmb");
	WriteBinMat(normal_path, normal);
	path weak_path = problem.result_folder / path("weak.bin");
	WriteBinMat(weak_path, pixel_states);
	// Medida: photometric confidence source (OpenMVS stores 1 - NCC cost).
	WriteBinMat(problem.result_folder / path("costs.dmb"), APD.GetCosts());
	path selected_view_path = problem.result_folder / path("selected_views.bin");
	WriteBinMat(selected_view_path, APD.GetSelectedViews());
	if (problem.params.use_radius) {
		path radius_path = problem.result_folder / path("radius.bin");
		WriteBinMat(radius_path, APD.GetRadiusMap());
	}

	if (problem.iteration == 15) {
		path depths_geom = problem.result_folder / path("depths_geom.dmb");
		writeDepthDmb(depths_geom, depth);
		path normals_path = problem.result_folder / path("normals.dmb");
		writeNormalDmb(normals_path, normal);
		path weak_img_path_2 = problem.result_folder / path("weak.png");
		ShowWeakImage(weak_img_path_2, pixel_states);
	}


	//yzl
	//for (int i = 0; i < problem.src_image_ids.size(); ++i) {
	//	std::string view_name = "view_" + std::to_string(i) + ".png";
	//	path view_path = problem.result_folder / path(view_name);
	//	/*std::cout << "view_path.string(): " << view_path.string() << std::endl;*/
	//	cv::imwrite(view_path.string(), matVector[i]);
	//}

	if (problem.show_medium_result) {
		path depth_img_path = problem.result_folder / path("depth_" + std::to_string(problem.iteration) + ".jpg");
		path normal_img_path = problem.result_folder / path("normal_" + std::to_string(problem.iteration) + ".jpg");
		path weak_img_path = problem.result_folder / path("weak_" + std::to_string(problem.iteration) + ".jpg");
		ShowDepthMap(depth_img_path, depth, APD.GetDepthMin(), APD.GetDepthMax());
		ShowNormalMap(normal_img_path, normal);
		ShowWeakImage(weak_img_path, pixel_states);

		// if ((problem.iteration + 1) % 4 == 0) {
		// 	path image_folder = problem.dense_folder / path("images");
		// 	path cam_folder = problem.dense_folder / path("cams");
		// 	path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		// 	path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		// 	path point_cloud_path = problem.result_folder / path("point_" + std::to_string(problem.iteration) + ".ply");
		// 	// path point_cloud_path = problem.result_folder / path("point_test_" + std::to_string(problem.iteration) + ".ply");

		// 	// for (int r = 0; r < height; ++r) for (int c = 0; c < width; ++c) if (pixel_states.at<uchar>(r, c) != STRONG) depth.at<float>(r, c) = 0;
		// 	ExportDepthImagePointCloud(point_cloud_path, image_path, cam_path, depth, APD.GetDepthMin(), APD.GetDepthMax());
		// }
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
	std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

static void PrintUsage() {
	std::cerr << "USAGE: APD dense_folder [--gpu N] [--final-scale N] [--show-medium]\n"
	          << "  --final-scale N   coarsest-to-finest schedule stops at 1/N image resolution\n"
	          << "                    (upstream behaviour is 2; 1 = full resolution)\n"
	          << "  --show-medium     write per-iteration debug jpgs into APD/<id>/\n"
	          << "  --fusion-only     skip PatchMatch; re-run fusion on existing APD/<id>/ results\n"
	          << "  --min-fuse-views N  source views that must agree for a fused point (upstream 1)\n"
	          << "  --fuse-reproj-px X    max reprojection error for two views to agree (upstream 2)\n"
	          << "  --fuse-depth-diff X   max relative depth difference to agree (upstream 0.01)\n"
	          << "  --fuse-normal-deg X   max normal angle to agree; <= 0 disables the test (upstream 10)\n";
}

int main(int argc, char** argv) {
	if (argc < 2) {
		PrintUsage();
		return EXIT_FAILURE;
	}
	path dense_folder(argv[1]);
	int gpu_index = 0;
	int final_scale = 2;
	bool show_medium = false;
	bool fusion_only = false;
	int min_fuse_views = 1;
	FusionThresholds fuse_th;
	for (int a = 2; a < argc; ++a) {
		std::string arg(argv[a]);
		if (arg == "--gpu" && a + 1 < argc) {
			gpu_index = std::atoi(argv[++a]);
		}
		else if (arg == "--final-scale" && a + 1 < argc) {
			final_scale = std::atoi(argv[++a]);
		}
		else if (arg == "--show-medium") {
			show_medium = true;
		}
		else if (arg == "--fusion-only") {
			fusion_only = true;
		}
		else if (arg == "--min-fuse-views" && a + 1 < argc) {
			min_fuse_views = std::atoi(argv[++a]);
		}
		else if (arg == "--fuse-reproj-px" && a + 1 < argc) {
			fuse_th.max_reproj_px = std::atof(argv[++a]);
		}
		else if (arg == "--fuse-depth-diff" && a + 1 < argc) {
			fuse_th.max_rel_depth_diff = std::atof(argv[++a]);
		}
		else if (arg == "--fuse-normal-deg" && a + 1 < argc) {
			const double deg = std::atof(argv[++a]);
			fuse_th.max_normal_rad = deg > 0 ? deg * M_PI / 180.0 : std::numeric_limits<float>::infinity();
		}
		else {
			PrintUsage();
			return EXIT_FAILURE;
		}
	}
	if (final_scale < 1 || (final_scale & (final_scale - 1)) != 0) {
		std::cerr << "--final-scale must be a power of two >= 1\n";
		return EXIT_FAILURE;
	}
	if (min_fuse_views < 1) {
		std::cerr << "--min-fuse-views must be >= 1\n";
		return EXIT_FAILURE;
	}
	path output_folder = dense_folder / path("APD");
	create_directory(output_folder);
	cudaSetDevice(gpu_index);
	// generate problems
	std::vector<Problem> problems;
	GenerateSampleList(dense_folder, problems);
	for (auto& problem : problems) {
		problem.show_medium_result = show_medium;
	}
	int num_images = problems.size();
	std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

	int round_num = ComputeRoundNum(problems);
	// Upstream runs scales 2^(round_num-1) ... 2 and never the full-resolution
	// round; --final-scale 1 adds it.
	int final_scale_log2 = 0;
	while ((1 << final_scale_log2) < final_scale) final_scale_log2++;
	const int rounds_to_run = std::max(1, round_num - final_scale_log2);

	std::cout << "Round nums: " << round_num << ", running " << rounds_to_run << " round(s) down to scale " << final_scale << std::endl;
	int iteration_index = 0;
	bool flag = true;
	for (int i = 0; i < rounds_to_run && !fusion_only; ++i) {
		const auto round_start = std::chrono::steady_clock::now();
		/*if(params->)*/
		for (auto& problem : problems) {
			problem.iteration = iteration_index;
			// 下面还有
			if (problem.ref_image_id == 93 && problem.iteration == 13) flag = true;
			problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale 
			{
				auto& params = problem.params;
				if (i == 0) {
					params.state = FIRST_INIT;
					params.use_APD = false;
				}
				else {
					params.state = REFINE_INIT;
					params.use_APD = true;
					params.ransac_threshold = 0.01 - i * 0.00125;
					params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
					if (i < round_num - 1) {
						params.use_detail = true;
					}
					else {
						params.use_detail = false;
					}
				}
				params.geom_consistency = false;
				params.max_iterations = 3;
				params.weak_peak_radius = 6;
			}
			if (flag) {
				GetProblemEdges(problem); // 注意要先得到 scale_size
				ProcessProblem(problem);
			}
		}
		iteration_index++;
		for (int j = 0; j < 3; ++j) {
			for (auto& problem : problems) {
				problem.iteration = iteration_index;
				if (problem.ref_image_id == 93 && problem.iteration == 13) flag = true;
				problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale 
				{
					auto& params = problem.params;
					params.state = REFINE_ITER;
					if (i == 0) {
						params.use_APD = false;
					}
					else {
						params.use_APD = true;
					}
					params.ransac_threshold = 0.01 - i * 0.00125;
					params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
					params.geom_consistency = true;
					params.max_iterations = 3;
					params.weak_peak_radius = MAX(4 - 2 * j, 2);
				}
				if (flag) {
					ProcessProblem(problem);
				}
			}
			iteration_index++;
		}
		const double round_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - round_start).count();
		std::cout << "Round: " << i << " done (scale " << problems[0].scale_size << ", " << round_s << " s)" << std::endl;
	}

	const auto fusion_start = std::chrono::steady_clock::now();
	RunFusion(dense_folder, problems, min_fuse_views, fuse_th);
	std::cout << "Fusion done (" << std::chrono::duration<double>(std::chrono::steady_clock::now() - fusion_start).count() << " s)" << std::endl;
	// {// delete files
	// 	for (size_t i = 0; i < problems.size(); ++i) {
	// 		const auto &problem = problems[i];
	// 		remove(problem.result_folder / path("weak.bin"));
	// 		remove(problem.result_folder / path("depths.dmb"));
	// 		remove(problem.result_folder / path("APD_normals.dmb"));
	// 		remove(problem.result_folder / path("selected_views.bin"));
	// 		remove(problem.result_folder / path("neighbour.bin")); 
	// 		remove(problem.result_folder / path("neighbour_map.bin"));
	// 	}
	// }
	std::cout << "All done\n";
	return EXIT_SUCCESS;
}
