#pragma once

#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>
#include <memory>

#include <opencv2/core/core.hpp>


std::vector<uchar> matToBytes(const cv::Mat &mat) {
    std::vector<uchar> bytes;
    if (mat.isContinuous()) {
        bytes.assign(mat.data, mat.data + mat.total() * mat.elemSize());
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            bytes.insert(bytes.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i) + mat.cols*mat.elemSize());
        }
    }
    return bytes;
}

// Serialization
template <class Archive>
void serialize(Archive & archive, cv::Mat& mat) {
    int type = mat.type();
    int rows = mat.rows;
    int cols = mat.cols;
    auto data = matToBytes(mat); // Convert mat to bytes

    archive(cereal::make_nvp("type", type),
            cereal::make_nvp("rows", rows),
            cereal::make_nvp("cols", cols),
            cereal::make_nvp("data", data)); // Serialize data
}

namespace cv {
	template<class Archive, class T>
	void serialize(Archive & archive, cv::Mat_<T> & mat)
	{
		int rows, cols, type;
		bool continuous;

		rows = mat.rows;
		cols = mat.cols;
		type = mat.type();
		continuous = mat.isContinuous();

		/*
		if (rows == 0 || cols == 0) {			
			archive.finishNode();
			return;
		}
		*/
		
		archive & cereal::make_nvp("dims", mat.dims);
		archive & cereal::make_nvp("rows", rows);
		archive & cereal::make_nvp("cols", cols);
		//ar & cereal::make_nvp("dims", mat.size());

		if (continuous) 
		{
			archive.setNextName("data");
			archive.startNode();
			archive.makeArray();
			
			for (auto && it : mat)
			{
				archive(it);
			}
			archive.finishNode();
		}
	}

	template<class Archive>
	void serialize(Archive & archive, cv::DMatch & o)
	{
		archive(CEREAL_NVP(o.queryIdx),
			CEREAL_NVP(o.trainIdx),
			CEREAL_NVP(o.imgIdx),
			CEREAL_NVP(o.distance)
		);
	}

	template<class Archive>
	void serialize(Archive & archive, cv::Size &o)
	{
		archive(CEREAL_NVP(o.height),
			CEREAL_NVP(o.width)
		);
	}

	template<class Archive>
	void serialize(Archive & archive, cv::Point2f &o)
	{
		archive(CEREAL_NVP(o.x),
			CEREAL_NVP(o.y)
		);
	}

	template<class Archive>
	void serialize(Archive & archive, cv::KeyPoint &o)
	{
		archive(CEREAL_NVP(o.pt),
			CEREAL_NVP(o.angle),
			CEREAL_NVP(o.size),
			CEREAL_NVP(o.octave)
		);
	}



	/**
	 * Serialise a cv::Mat using cereal.
	 *
	 * Supports all types of matrices as well as non-contiguous ones.
	 *
	 * @param[in] ar The archive to serialise to.
	 * @param[in] mat The matrix to serialise.
	 */
	template<class Archive>
	void save(Archive& ar, const cv::Mat& mat)
	{
		int rows, cols, type;
		bool continuous;

		rows = mat.rows;
		cols = mat.cols;
		type = mat.type();
		continuous = mat.isContinuous();

		ar & rows & cols & type & continuous;

		if (continuous) {
			const int data_size = rows * cols * static_cast<int>(mat.elemSize());
			auto mat_data = cereal::binary_data(mat.ptr(), data_size);
			ar & mat_data;
		}
		else {
			const int row_size = cols * static_cast<int>(mat.elemSize());
			for (int i = 0; i < rows; i++) {
				auto row_data = cereal::binary_data(mat.ptr(i), row_size);
				ar & row_data;
			}
		}
	};

	/**
	 * De-serialise a cv::Mat using cereal.
	 *
	 * Supports all types of matrices as well as non-contiguous ones.
	 *
	 * @param[in] ar The archive to deserialise from.
	 * @param[in] mat The matrix to deserialise into.
	 */
	template<class Archive>
	void load(Archive& ar, cv::Mat& mat)
	{
		int rows, cols, type;
		bool continuous;

		ar & rows & cols & type & continuous;

		if (continuous) {
			mat.create(rows, cols, type);
			const int data_size = rows * cols * static_cast<int>(mat.elemSize());
			auto mat_data = cereal::binary_data(mat.ptr(), data_size);
			ar & mat_data;
		}
		else {
			mat.create(rows, cols, type);
			const int row_size = cols * static_cast<int>(mat.elemSize());
			for (int i = 0; i < rows; i++) {
				auto row_data = cereal::binary_data(mat.ptr(i), row_size);
				ar & row_data;
			}
		}
	};
}

void saveKeyPointsToCSV(const std::map<int, cv::KeyPoint>& keyPointsMap, const std::string& filename) {
	std::ofstream file(filename);

	// Check if the file is open
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	// Write the CSV header
	file << "ID,x,y,size,angle,response,octave,class_id\n";

	// Iterate over the map and write each keypoint's attributes to the file
	for (const auto& pair : keyPointsMap) {
		const auto& id = pair.first;
		const auto& kp = pair.second;
		file << id << ","
			<< kp.pt.x << ","
			<< kp.pt.y << ","
			<< kp.size << ","
			<< kp.angle << ","
			<< kp.response << ","
			<< kp.octave << ","
			<< kp.class_id << "\n";
	}

	// Close the file
	file.close();
}