/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <rtabmap/core/Odometry.h>
#include "rtabmap/core/Rtabmap.h"
#include "rtabmap/core/RtabmapThread.h"
#include "rtabmap/core/CameraRGBD.h"
#include "rtabmap/core/CameraStereo.h"
#include "rtabmap/core/CameraThread.h"
#include "rtabmap/core/OdometryThread.h"
#include "rtabmap/core/Graph.h"
#include "rtabmap/utilite/UEventsManager.h"
#include <QApplication>
#include <stdio.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/filter.h>

#include "rtabmap/core/Memory.h"
#include "rtabmap/core/VWDictionary.h"
#include "rtabmap/core/VisualWord.h"
#include "rtabmap/core/Cerealize.h"

#include "tqdm/tqdm.h"

#ifdef RTABMAP_PYTHON
#include "rtabmap/core/PythonInterface.h"
#endif

using namespace rtabmap;
using namespace cv;

int load_loop_closures(const std::string &path2file, 
	std::vector<std::pair<int, float>> &data) 
{
    
	std::ifstream file(path2file);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return 1;
    }

    int key;
    float value;
    char delimiter;

    while (file >> key >> delimiter >> value) {
        if (delimiter == ',') {
            data.push_back(std::make_pair(key, value));
        }
    }

    file.close();
    return 0;
}


std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}


bool has_extension(const std::string& file, const std::vector<std::string>& exts) {
    for (const auto &ext : exts) 
	{
		if (file.length() >= ext.length()) {
			if (0 == file.compare(file.length() - ext.length(), ext.length(), ext)){
				return true;
			}
		} 
		else {
			continue;
		}
	}
	return false;
}

void get_files(const std::string &path, std::vector<std::string> &files_in_dir, std::vector<std::string> extension={".png", ".jpg", ".tif", ".bmp"})
{
	DIR *dir;
    struct dirent *ent;
	
	//std::vector<std::string> extension = {".yml"};
	//std::string path = "data/samples/"; // Change this to your directory path
    //std::vector<std::string> extension = ; // Change this to the desired extension

	if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string file_name = ent->d_name;
            if (has_extension(file_name, extension)) {
                //std::cout << file_name << std::endl;
				files_in_dir.push_back(path + file_name);
            }
        }
        closedir(dir);
	}
}

/**
 * Compute the likelihood of the signature with some others in the memory.
 * Important: Assuming that all other ids are under 'signature' id.
 * If an error occurs, the result is empty.
 * ids - list of signature ids to compare
 */
std::map<int, float> computeLikelihood(VWDictionary* vwd, 
	const std::list<int> &ids, 
	const std::list<int> &wordIds,
	std::map<int, float> &wordCount,
	float N)
{
	
	{
		UTimer timer;
		timer.start();
		std::map<int, float> likelihood;
		std::map<int, float> calculatedWordsRatio;

		for(std::list<int>::const_iterator iter = ids.begin(); iter!=ids.end(); ++iter)
		{
			likelihood.insert(likelihood.end(), std::pair<int, float>(*iter, 0.0f));
		}

		//const std::list<int> & wordIds = uUniqueKeys(signature->getWords());
		

		float nwi; // nwi is the number of a specific word referenced by a signature
		float ni; // ni is the total of words referenced by a signature
		float nw; // nw is the number of signatures referenced by a specific word
		//float N; // N is the total number of places

		float logNnw;
		const VisualWord * vw;
		
		UDEBUG("processing... ");
		//run on all words in the image

		std::cout << ids.size() << std::endl;
		for(std::list<int>::const_iterator i=wordIds.begin(); i!=wordIds.end(); ++i)
		{
			if(*i>0)
			{
				// Compute score TF-IDF
				// Get word from dictionary
				vw = vwd->getWord(*i);
				UASSERT_MSG(vw!=0, uFormat("Word %d not found in dictionary!?", *i).c_str());

				//(signature id , occurrence in the signature)
				const std::map<int, int> & refs = vw->getReferences();
				nw = refs.size();
				if(nw)
				{
					logNnw = log10(N/nw);
					if(logNnw)
					{
						for(std::map<int, int>::const_iterator j=refs.begin(); j!=refs.end(); ++j)
						{
							std::map<int, float>::iterator iter = likelihood.find(j->first);
							if(iter != likelihood.end())
							{
								nwi = j->second;
								ni = wordCount[j->first];
								//ni = this->getNi(j->first);
								if(ni != 0)
								{
									//UDEBUG("%d, %f %f %f %f", vw->id(), logNnw, nwi, ni, ( nwi  * logNnw ) / ni);
									iter->second += ( nwi  * logNnw ) / ni;
								}
							}
						}
					}
				}
			}
		}
		UDEBUG("compute likelihood (tf-idf) %f s", timer.ticks());
		std::cout << "compute likelihood (tf-idf) " << timer.ticks() << std::endl;
		return likelihood;
	}
}

cv::Mat load_descriptors(const std::string &file_path)
{
    // Create a FileStorage object for reading
    cv::FileStorage file_storage(file_path, cv::FileStorage::READ);

    // Read the descriptors
    cv::Mat descriptors;
    file_storage["desc"] >> descriptors;

    // Release the file
    file_storage.release();

    return descriptors;
}

std::map<int,std::string> convert_files2map(const std::vector<std::string> &paths)
{
	std::map<int, std::string> out;

	for (auto path: paths)
	{
		std::size_t lastSlashPos = path.find_last_of("desc");

		// Find the last '.' character
		std::size_t lastDotPos = path.find_last_of(".");

		// Extract the "id" between the last '/' and the last '.'
		
		if (lastSlashPos != std::string::npos && lastDotPos != std::string::npos && lastDotPos > lastSlashPos) {
			int id;
			id = std::atoi(&path.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1)[0]);
			out[id] = path;
		}
	}
	return out;
}

std::vector<int> read_nonduplicates(const std::string &fname)
{
	std::ifstream file(fname);
	std::vector<int> numbers;
    
	if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return numbers;
    }

    int number;
    while (file >> number) {
        numbers.push_back(number);
    }

    file.close();
	return numbers;
}

void train_incremental(
	const std::vector<std::string> &dataset_files,
	const std::string &fileNameReferences,
	const std::string &fileNameDescriptors,
	bool detect_describe=true)
{

	ParametersMap params;
	Rtabmap * rtabmap = new Rtabmap();
	rtabmap->init(params);
	Memory* memory = rtabmap->getMemoryC();
	VWDictionary* vwd = memory->getVWDictionaryC(); 

	std::map<int, std::string> path_map = convert_files2map(dataset_files);
	vwd->setIncrementalDictionary();

	std::vector<int> signature_ids = read_nonduplicates("duplicates_20240221_072415536.csv");
	
	// The paths are automatically sorted by their IDs due to the nature of std::map
    // Iterate over the map

	//for (const auto& pair : path_map)
	size_t N = signature_ids.size();
	for(int id : tqdm::range(N))
	{
		//int id = pair.first;
		// if (id%3!=0){
		// 	continue;
		// }
		//std::string f = pair.second;
		int id_ = signature_ids[id];
		std::string f = path_map[id_];
		//std::cout << f << std::endl;

		cv::Mat features;
		
		if(detect_describe){
			cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
			cv::Mat im = cv::imread(f);
			std::vector<cv::KeyPoint> keypoints;
			orb->detect(im, keypoints);
			orb->compute(im, keypoints, features);
		}
		else{
			features = load_descriptors(f);
		}
		vwd->update();
		std::list<int> words = vwd->addNewWords(features, id_);
		// if(id >= 225){
		// 	break;
		// }
		id++;
	}
	vwd->update();
	vwd->deleteUnusedWords();
	std::cout << "Number of words:" << vwd->getIndexedWordsCount() << std::endl;
	vwd->exportDictionary(&fileNameReferences[0], &fileNameDescriptors[0]);
	rtabmap->close(false);
	return;
}

std::map<int, float> test_rtabmap(
	const std::string &fileNameDescriptors, 
	std::string &database_path,
	const int key_id,
	bool detect_describe=true)
{

	ParametersMap params;
	Rtabmap * rtabmap = new Rtabmap();
	rtabmap->init(params);
	Memory* memory = rtabmap->getMemoryC();
	VWDictionary* vwd = memory->getVWDictionaryC(); 
	vwd->setFixedDictionary(&fileNameDescriptors[0]);
	std::cout << "Number of words:" << vwd->getIndexedWordsCount() << std::endl;

	std::vector<std::pair<int, float>> scores;

	std::string path2file = database_path + "/" + std::to_string(key_id) + ".csv";
	load_loop_closures(path2file, scores);

	std::map<int, std::list<int>> wordIds;
	std::map<int, float> wordC;

	std::list<int> query_ids;
	for (const auto &s: scores)
	{
		int sid = s.first;
		std::string f = database_path + "/desc" + std::to_string(sid) + ".yml";
		cv::Mat features;
		if(detect_describe){
			cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
			cv::Mat im = cv::imread(f);
			std::vector<cv::KeyPoint> keypoints;
			orb->detect(im, keypoints);
			orb->compute(im, keypoints, features);
		}
		else{
			features = load_descriptors(f);
		}
		wordIds[sid] = vwd->addNewWords(features, sid);
		//vwd->update();
		wordC[sid] = wordIds[sid].size();
		query_ids.push_back(sid);
	}

	std::string f = database_path + "/desc" + std::to_string(key_id) + ".yml";
	cv::Mat features;
	if(detect_describe){
		cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
		cv::Mat im = cv::imread(f);
		std::vector<cv::KeyPoint> keypoints;
		orb->detect(im, keypoints);
		orb->compute(im, keypoints, features);
	}
	else{
		features = load_descriptors(f);
	}

	wordIds[key_id] = vwd->addNewWords(features, key_id);

	int N = wordC.size();
	auto start_time = std::chrono::high_resolution_clock::now();
	std::map<int, float> likelihood = computeLikelihood(vwd, query_ids, wordIds[key_id], wordC, N);
	auto end_time = std::chrono::high_resolution_clock::now();
	// Calculate the elapsed time in milliseconds
    auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Elapsed time: " << elapsed_time_ms << " milliseconds" << std::endl;
	rtabmap->close(false);
	return likelihood;
}


std::map<int, float> test_fixed(
	const std::string &fileNameDescriptors, 
	const std::vector<std::string> &dataset_files,
	std::list<int> &query_ids, 
	const int key_id,
	bool detect_describe=true)
{

	ParametersMap params;
	Rtabmap * rtabmap = new Rtabmap();
	rtabmap->init(params);
	Memory* memory = rtabmap->getMemoryC();
	VWDictionary* vwd = memory->getVWDictionaryC(); 
	vwd->setFixedDictionary(&fileNameDescriptors[0]);
	std::cout << "Number of words:" << vwd->getIndexedWordsCount() << std::endl;

	std::map<int, std::list<int>> wordIds;
	std::map<int, float> wordC;
	std::vector<cv::Mat> features_all(dataset_files.size());
	
	int id = 0;
	for (const auto &f: dataset_files)
	{
		cv::Mat features;
		if(detect_describe){
			cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
			cv::Mat im = cv::imread(f);
			std::vector<cv::KeyPoint> keypoints;
			orb->detect(im, keypoints);
			orb->compute(im, keypoints, features);
		}
		else{
			features = load_descriptors(f);
		}
		wordIds[id] = vwd->addNewWords(features, id);
		vwd->update();
		wordC[id] = wordIds[id].size();
		features_all[id] = features;
		id++;
	}

	int N = wordC.size();
	std::map<int, float> likelihood = computeLikelihood(vwd, query_ids, wordIds[key_id], wordC, N);
	rtabmap->close(false);
	return likelihood;
}

int main(int argc, char * argv[])
{
	std::string fileNameReferences = "ref.txt";
	std::string fileNameDescriptors = "DictionaryLC4large_online.txt";
	
	if (true){
		std::string database_path = "/home/gvasserm/dev/aicv_amr_ws/results_lc4large_map_def/";
		int key_id = 305;
		std::map<int, float> likelihood = test_rtabmap(fileNameDescriptors, database_path, key_id, false);
		cerealizeLikelihood(std::to_string(key_id) + "fixed.csv", likelihood);
		return -1;
	}

	if(false){
		std::string data_path = "results_gftt_default_ptk/";
		std::vector<std::string> dataset_files;
		std::vector<std::string> extension ={".yml"};
		get_files(data_path, dataset_files, extension);
		std::sort(dataset_files.begin(), dataset_files.end());
	
		train_incremental(
			dataset_files,
			fileNameReferences,
			fileNameDescriptors, 
			false);
	}
	
	return 0;
}