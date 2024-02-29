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

#include "DBoW3.h"

#ifdef RTABMAP_PYTHON
#include "rtabmap/core/PythonInterface.h"
#endif

using namespace rtabmap;
using namespace cv;


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

void get_files(std::vector<std::string> &files_in_dir)
{
	DIR *dir;
    struct dirent *ent;
    std::string path = "/home/gvasserm/Downloads/Bicocca_Static_Lamps/temp/"; // Change this to your directory path
    std::vector<std::string> extension = {".png", ".jpg", ".tif", ".bmp"}; // Change this to the desired extension

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
		return likelihood;
	}
}


int main(int argc, char * argv[])
{
	ParametersMap params;
	//param.insert(ParametersPair(Parameters::kRGBDCreateOccupancyGrid(), "true")); // uncomment to create local occupancy grids

	// Create RTAB-Map to process OdometryEvent
	std::vector<std::string> files_in_dir;
	get_files(files_in_dir);
	std::sort(files_in_dir.begin(), files_in_dir.end());

	int id = 0;
	Rtabmap * rtabmap = new Rtabmap();
	rtabmap->init(params);
	Memory* memory = rtabmap->getMemoryC();
	VWDictionary* vwd = memory->getVWDictionaryC();

	//vwd->setFixedDictionaryDBOW2("/home/gvasserm/dev/ORB_SLAM2/Vocabulary/ORBvoc.txt");

	std::map<int, std::list<int>> wordIds;
	std::map<int, float> wordC;

	if (true)
	{
		for (const auto &f : files_in_dir)
		{

			if (id > 0)
			{
				cv::Mat im = cv::imread(f);
				cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
				std::vector<cv::KeyPoint> keypoints;
				cv::Mat features;
				orb->detect(im, keypoints);
				orb->compute(im, keypoints, features);
				std::list<int> wi = vwd->addNewWords(features, id);
				// std::vector<int> w = vwd->findNN(features);
				vwd->update();
				// wordIds[id] = wi;
				// wordC[id] = wi.size();
				// int s = vwd->getVisualWords().size();

				if (false)
				{
					cv::Mat image_with_keypoints;
					drawKeypoints(im, keypoints, image_with_keypoints);

					// Display the original image and the one with keypoints
					imshow("Original Image", im);
					imshow("Image with Keypoints", image_with_keypoints);
					waitKey(0);
					// std::cout << features.rows << std::endl;
					// std::cout << s << std::endl;
					// s = vwd->getVisualWords().size();
					// std::cout << s << std::endl;
				}
			}
			id++;
		}
	}

	//std::cout << wordC[0] << std::endl;

	std::map<int, DBoW3::BowVector> bowVectors;
	if (true)
	{
		vwd->setFixedDictionary();
		id = 0;
		for (const auto &f : files_in_dir) 
		{
			cv::Mat im = cv::imread(files_in_dir[id]);
			cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
			std::vector<cv::KeyPoint> keypoints;
			cv::Mat features;
			orb->detect(im, keypoints);
			orb->compute(im, keypoints, features);
			wordIds[id] = vwd->addNewWords(features, id);
			wordC[id] = wordIds[id].size();

			DBoW3::BowVector mBowVec;
    		DBoW3::FeatureVector mFeatVec;

			std::vector<cv::Mat> vCurrentDesc = toDescriptorVector(features);
			memory->_vocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
			bowVectors[id] = mBowVec;
			
			// std::vector<int> wi = vwd->findNN(features);
			// std::list<int> lst(wi.begin(), wi.end());
			// wordIds[id] = lst;
			// wordC[id] = wordIds[id].size();
			// vwd->update();
			id++;
		}
	}

	int N = wordC.size();

	std::list<int> ids = {1, 3, 5, 29};
	std::map<int, float> likelihood = computeLikelihood(vwd, ids, wordIds[0], wordC, N);
	
	std::map<int, double> scores;
	for(std::list<int>::const_iterator i=ids.begin(); i!=ids.end(); ++i)
	{
		DBoW3::BowVector BowVec1 = bowVectors[0];
		DBoW3::BowVector BowVec2 = bowVectors[*i];
		scores[*i] = memory->_vocabulary->score(BowVec1, BowVec2);
	}
	
	rtabmap->close(false);
	return 0;
}
