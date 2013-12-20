/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */


// Stdlib
#include <stdlib.h>
#include <cmath>
#include <limits.h>

// PCL input/output
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/visualization/pcl_visualizer.h>

//PCL other
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/supervoxel_clustering.h>

// The segmentation class this example is for
#include <pcl/segmentation/lccp_segmentation.h>

// VTK
#include <vtkImageReader2Factory.h>
#include <vtkImageReader2.h>
#include <vtkImageData.h>
#include <vtkImageFlip.h>
#include <vtkPolyLine.h>


/// *****  Type Definitions ***** ///

typedef pcl::PointXYZRGBA PointT; // The point type used for input

typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList;


/// Callback and variables

bool show_normals = false, normals_added = false;
bool show_adjacency = false;
bool show_supervoxels = false;
bool show_help = true;


/** \brief Callback for setting options in the visualizer via keyboard.
 *  \param[in] event Registered keyboard event
 */
void
keyboard_callback (const pcl::visualization::KeyboardEvent& event, void*)
{
  int key = event.getKeyCode ();

  if (event.keyUp ())
    switch (key)
    {
      case (int)'1': show_normals = !show_normals; break;
      case (int)'2': show_adjacency = !show_adjacency; break;
      case (int)'3': show_supervoxels = !show_supervoxels; break;
      case (int)'h': case (int)'H': show_help = !show_help; break;
      default: break;
    }
}


/// *****  Prototypes ***** ///


/** \brief Generates a pointcloud from two PNG Files, one containing the RGB and the other containing the depth data.
 *  \param[in] rgb_path Path to PNG-File with RGB-data.
 *  \param[in] depth_path Path to PNG-File with depth-data.
 *  \param[out] output_cloud Generated pointcloud */
void
rgbd2pointcloud(std::string rgb_path, std::string depth_path, pcl::PointCloud<PointT>::Ptr output_cloud);

/** \brief Fill vector with random colors
 *  \param[in] label_colors Vector to be filled.  */
void
drawColors(std::vector< uint32_t >& label_colors, uint nr_labels);

/** \brief Displays info text in the specified PCLVisualizer
 *  \param[in] viewer The PCLVisualizer to modifz  */
void
printText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

/** \brief Removes info text in the specified PCLVisualizer
 *  \param[in] viewer The PCLVisualizer to modifz  */
void
removeText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);



/// ---- main ---- ///

int
main(int argc, char ** argv)
{

  if (argc < 2) /// Print Info
  {
    pcl::console::print_info (\
"\n\
-- pcl::LCCPSegmentation example -- :\n\
\n\
(-p <pcd-file>) or (-pn <pcd-file with normals>) or ((-r <rgb_file> and -d <depth_file>) -  Input to use.\n\
-o <outname> -  Write output files to disk. If this option is specified without giving a name, the outputname defaults to the <inputfilename>.\n\
                The following files are written:\n\
                  <outname>_out.pcd - Labeled point cloud \n\
                  <outname>_out.png - Colored PNG file (only for organized input clouds) \n\
                  <outname>_seglabel.png - One channel 16-bit PNG with labels (only for organized input clouds) \n\
    -so - Write colored Supervoxel image to file <outfilename>_sv.png and point cloud to <outfilename>_svcloud.pcd\n\
    -novis - Disable visualization.\n\
    \n\
 Supervoxel Parameters: \n\
    -v <voxel resolution> \n\
    -s <seed resolution> \n\
    -c <color weight> \n\
    -z <spatial weight> \n\
    -n <normal_weight> \n\
    -tvoxel - Use single-camera-transform for voxels \n\
    -refine - Use supervoxel refinement\n\
 LCCPSegmentation Parameters: \n\
    -ct - <concavity tolerance angle> \n\
    -ec - use extended (less local) convexity check\n\
    -smooth <filter size>  - filter the resulting image\n\
    \n", argv[0]);
    return (1);
  }


  /// -----------------------------------|  Preparations  |-----------------------------------

  bool sv_output_specified = pcl::console::find_switch (argc, argv, "-so");
  bool show_visualization = (not pcl::console::find_switch (argc, argv, "-novis"));


/// Create variables needed for preparations
  std::string inputname(""), outputname("");
  pcl::PointCloud<PointT>::Ptr input_CloudPtr(new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_rgbNormalCloudPtr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::Normal>::Ptr input_normalsPtr(new pcl::PointCloud<pcl::Normal>);


/// Get path of rgb+d files if specified
  std::string rgb_path;
  bool rgb_file_specified = pcl::console::find_switch (argc, argv, "-r");
  if (rgb_file_specified)
  {
    pcl::console::parse (argc, argv, "-r", rgb_path);
    inputname = rgb_path;
  }

  std::string depth_path;
  bool depth_file_specified = pcl::console::find_switch (argc, argv, "-d");
  if (depth_file_specified)
    pcl::console::parse (argc, argv, "-d", depth_path);


/// Get pcd path if specified
  std::string pcd_path;
  bool rgba_cloud_specified = pcl::console::find_switch (argc, argv, "-p");
  bool rgbNormals_cloud_specified = pcl::console::find_switch (argc, argv, "-pn");

  bool pcd_file_specified = (rgba_cloud_specified  or  rgbNormals_cloud_specified);
  if (pcd_file_specified)
  {
    if(rgba_cloud_specified)
    {
      pcl::console::parse (argc,argv,"-p",pcd_path);
      inputname = pcd_path;
    }
    if(rgbNormals_cloud_specified)
    {
      pcl::console::parse (argc,argv,"-pn",pcd_path);
      inputname = pcd_path;
    }
  }


/// Abort if no rgb/depth pair or input cloud was given
  if (  (!rgb_file_specified || !depth_file_specified) and  !pcd_file_specified)
  {
    std::cout << "No cloud specified!" << std::endl;
    return (1);
  }


/// If no pcd was given, create cloud out of rgb/depth pair
  if (!pcd_file_specified)
  {
    rgbd2pointcloud(rgb_path, depth_path, input_CloudPtr);
  }
  else
  {
    std::cout << "Loading pointcloud...";

    if(rgba_cloud_specified)
    {
      pcl::io::loadPCDFile (pcd_path, *input_CloudPtr);

    }
    if(rgbNormals_cloud_specified)
    {
      pcl::io::loadPCDFile (pcd_path, *input_rgbNormalCloudPtr);
      pcl::copyPointCloud(*input_rgbNormalCloudPtr, *input_CloudPtr);
      pcl::copyPointCloud(*input_rgbNormalCloudPtr, *input_normalsPtr);

      //FIXME Supposedly there was a bug in old PCL versions that the orientation was not set correctly when recording clouds. This is just a workaround.
      if(input_normalsPtr->sensor_orientation_.w() == 0)
      {
        input_normalsPtr->sensor_orientation_.w() = 1;
        input_normalsPtr->sensor_orientation_.x() = 0;
        input_normalsPtr->sensor_orientation_.y() = 0;
        input_normalsPtr->sensor_orientation_.z() = 0;
      }
    }

    //FIXME Supposedly there was a bug in PCL that the orientation was not set correctly when recording clouds. This is just a workaround.
    if(input_CloudPtr->sensor_orientation_.w() == 0)
    {
      input_CloudPtr->sensor_orientation_.w() = 1;
      input_CloudPtr->sensor_orientation_.x() = 0;
      input_CloudPtr->sensor_orientation_.y() = 0;
      input_CloudPtr->sensor_orientation_.z() = 0;
    }
  }

std::cout << "Done making cloud!" << std::endl;


///  Create outputname if not given
bool output_specified = pcl::console::find_switch (argc, argv, "-o");
  if(output_specified)
  {
    pcl::console::parse (argc,argv,"-o",outputname);

    // If no filename is given, get output filename from inputname (strip seperators and file extension)
    if (outputname.empty() or (outputname.at(0) == '-') )
    {
      outputname = inputname;
      size_t sep = outputname.find_last_of('/');
      if(sep != std::string::npos)
        outputname = outputname.substr(sep+1, outputname.size()-sep-1);

      size_t dot = outputname.find_last_of('.');
      if(dot != std::string::npos)
        outputname = outputname.substr(0, dot);
    }
  }



/// -----------------------------------|  Main Computation  |-----------------------------------


  ///  Default values of parameters before parsing
    // Supervoxel Stuff
    float  voxel_resolution = 0.0075f;
    float  seed_resolution = 0.03f;
    float  color_importance = 0.0f;
    float  spatial_importance = 1.0f;
    float  normal_importance = 4.0f;
    bool   use_single_cam_transform = false;
    bool   use_supervoxel_refinement = false;

    // Segmentation Stuff
    float  concavity_tolerance_threshold = 10;
    uint32_t  filter_size = 0;
    bool use_extended_convexity = false;



  ///  Parse Arguments needed for computation
    //Supervoxel Stuff
    use_single_cam_transform = pcl::console::find_switch(argc, argv, "-tvoxel");
    use_supervoxel_refinement = pcl::console::find_switch(argc, argv, "-refine");

    pcl::console::parse(argc, argv, "-v", voxel_resolution);
    pcl::console::parse(argc, argv, "-s", seed_resolution);
    pcl::console::parse(argc, argv, "-c", color_importance);
    pcl::console::parse(argc, argv, "-z", spatial_importance);
    pcl::console::parse(argc, argv, "-n", normal_importance);

    // Segmentation Stuff
    pcl::console::parse(argc, argv, "-ct", concavity_tolerance_threshold);
    use_extended_convexity =  pcl::console::find_switch(argc, argv, "-ec");
    uint   k_factor = 0;
    if(use_extended_convexity) k_factor = 1;
    pcl::console::parse(argc, argv, "-smooth", filter_size);



  /// --------- Preparation of Input: Supervoxel Oversegmentation --------- ///

    pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution, use_single_cam_transform);
    super.setInputCloud (input_CloudPtr);
    if(rgbNormals_cloud_specified)
        super.setNormalCloud(input_normalsPtr);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);
    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;


    std::cout << "Extracting supervoxels!\n";
    super.extract (supervoxel_clusters);

    if(use_supervoxel_refinement)
    {
        std::cout << "Refining supervoxels \n";
        super.refineSupervoxels (2, supervoxel_clusters);
    }

    std::cout << "  Nr. Supervoxels: " << supervoxel_clusters.size() << std::endl;

    std::cout << "Getting supervoxel adjacency\n";
    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);


  /// Get the cloud of supervoxel centroid with normals and the colored cloud with supervoxel coloring (this is used for visulization)
    pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud( supervoxel_clusters );
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sv_colored_cloud = super.getColoredCloud(); // This is also used for supervoxel output




  /// --------- The Main Step:  Perform LCCPSegmentation --------- ///

    std::cout << "Starting Segmentation.. " << std::endl;
    pcl::LCCPSegmentation<PointT> svccSeg;
    svccSeg.setConcavityToleranceThreshold(concavity_tolerance_threshold);
    svccSeg.setSmoothnessCheck(true, voxel_resolution, seed_resolution);
    svccSeg.setKFactor(k_factor);

    svccSeg.segment(supervoxel_clusters, supervoxel_adjacency);

    if(filter_size>0)
    {
        std::cout << "  Noise filtering.." << std::endl;
        svccSeg.removeNoise(filter_size);

    }

    std::cout << "  Interpolation voxel cloud -> input cloud and relabeling.. " << std::endl;
    pcl::PointCloud<pcl::PointXYZL>::Ptr segment_labeled_cloud = super.getLabeledCloud();
    svccSeg.relabelCloud(segment_labeled_cloud);


    SuperVoxelAdjacencyList sv_adjacency_list = svccSeg.getSVAdjacencyList(); // Needed for visualization



  /// --- Creating Colored Clouds and Output ---
    if(segment_labeled_cloud->size() == input_CloudPtr->size() )
    {
      std::cout << "Coloring cloud.. " << std::endl;

      std::vector<uint32_t> label_colors;
      drawColors(label_colors, supervoxel_clusters.size());

      /* NOTE: segment_labeled_cloud was produced by getLabeledCloud(), which performs an interpolation to the input cloud.
          Thus, segment_labeled_cloud and the input cloud should have the same size and points with the same ID should correspond to one another. */
      for(int pointID=0; pointID<segment_labeled_cloud->size(); ++pointID )
      {
          PointT&  searchPoint = input_CloudPtr->points[pointID];
          pcl::PointXYZL&  labelPoint = segment_labeled_cloud->points[pointID];

          searchPoint.rgb = *reinterpret_cast<float*> (&label_colors[labelPoint.label]);
      }

      if(output_specified)
      {
        std::cout << "Saving output.. " << std::endl;

        pcl::io::savePCDFile(outputname + "_out.pcd", *segment_labeled_cloud);

        if(input_CloudPtr->isOrganized())
        {
            pcl::io::savePNGFile(outputname + "_out.png", *input_CloudPtr, "rgb");
            pcl::io::savePNGFile(outputname + "_seglabel.png", *segment_labeled_cloud, "label");

            if(sv_output_specified)
                pcl::io::savePNGFile(outputname + "_sv.png", *sv_colored_cloud, "rgb");
        }

        if(sv_output_specified)
        {
          pcl::io::savePCDFile(outputname + "_svcloud.pcd", *sv_centroid_normal_cloud);
        }
      }

    }
    else
    {
        PCL_ERROR("ERROR:: Sizes of input cloud and labeled supervoxel cloud do not match. No output is produced");
    }


  /// -----------------------------------|  Visualization  |-----------------------------------

  if(show_visualization)
  {

    /// Calculate visualization of adjacency graph
    // Using lines this would be VERY slow right now, because one actor is created for every line (may be fixed in future versions of PCL)
    // Currently this is a work-around creating a polygon mesh consisting of two triangles for each edge
    using namespace pcl;

    typedef LCCPSegmentation<PointT>::VertexIterator VertexIterator;
    typedef LCCPSegmentation<PointT>::AdjacencyIterator AdjacencyIterator;
    typedef LCCPSegmentation<PointT>::VertexID VertexID;
    typedef LCCPSegmentation<PointT>::EdgeID EdgeID;


    std::set<EdgeID> edge_drawn;
    //Note: Color format: 0x00RRGGBB;
    const uint32_t edge_color_convex = 0x00FFFFFF;
    const uint32_t edge_color_concave = 0x00FF0000;
    uint32_t edge_color;

      //The vertices in the supervoxel adjacency list are the supervoxel centroids
      //This iterates through them, finding the edges
      std::pair<VertexIterator, VertexIterator> vertex_iterator_range;
      vertex_iterator_range = boost::vertices(sv_adjacency_list);

    /// Create a cloud of the voxelcenters and map: VertexID in adjacency graph -> Point index in cloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr elevated_svcenters_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>); // This cloud is used to adress the (super)voxel centers via the adjacency list ID
    std::vector<pcl::Vertices> mesh_triangles;
    std::map<VertexID, uint> ID2cloudIDx;
    uint i=0;
    for (VertexIterator itr=vertex_iterator_range.first ; itr != vertex_iterator_range.second; ++itr)
    {
      const uint32_t sv_label = sv_adjacency_list[*itr];
      const pcl::PointXYZRGBA& sv_centroid = supervoxel_clusters[sv_label]->centroid_;
      const pcl::Normal& sv_normal = supervoxel_clusters[sv_label]->normal_;

      // Small displacement in direction of normal is needed, to elevate triangles a little bit above the surface.
      // This way, the triangles stay visible even if the point size is increased.
      pcl::PointXYZRGBA pt;
      pt.x = sv_centroid.x + 0.001 * sv_normal.normal_x;
      pt.y = sv_centroid.y + 0.001 * sv_normal.normal_y;
      pt.z = sv_centroid.z + 0.001 * sv_normal.normal_z;
      pt.rgba = 0;

      elevated_svcenters_cloud->push_back(pt);
      ID2cloudIDx[*itr] = i;
      ++i;
    }



    /// Loop through all Vertices and draw triangles to each neighbor
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr mesh_vertices (new pcl::PointCloud<pcl::PointXYZRGBA>); // This cloud is for the (colored) mesh vertices

    for (VertexIterator itr=vertex_iterator_range.first ; itr != vertex_iterator_range.second; ++itr)
    {
      const uint32_t sv_label = sv_adjacency_list[*itr];
      const pcl::PointXYZRGBA& sv_centroid = supervoxel_clusters[sv_label]->centroid_;

      std::pair<AdjacencyIterator, AdjacencyIterator> neighbors = boost::adjacent_vertices (*itr, sv_adjacency_list);

      for(AdjacencyIterator itr_neighbor = neighbors.first; itr_neighbor != neighbors.second; ++itr_neighbor)
      {

        EdgeID connecting_edge = boost::edge (*itr,*itr_neighbor, sv_adjacency_list).first;  //Get the edge connecting these supervoxels
        if( sv_adjacency_list[connecting_edge].isConvex ) edge_color = edge_color_convex;
        else edge_color = edge_color_concave;

        // // If the next line is uncommented, for each edge only one polygon will be drawn
        // if(!edge_drawn.count(connecting_edge)) edge_drawn.insert(connecting_edge);
        // else continue;


        pcl::PointXYZRGBA vert_curr = elevated_svcenters_cloud->points[ID2cloudIDx[*itr]];
        vert_curr.rgba = edge_color;
        mesh_vertices->push_back(vert_curr);

        pcl::PointXYZRGBA vert_neigh = elevated_svcenters_cloud->points[ID2cloudIDx[*itr_neighbor]];
        vert_neigh.rgba = edge_color;
        mesh_vertices->push_back(vert_neigh);


        // Two triangles are used to make the edge visible from all sides.
        // Triangle 1: Point - Neighbor - location of neighbor with small displacement in direction of its normal
        // Triangle 2: Point - Neighbor - location of neighbor with small displacement orthogonal to vector connecting point-neighbor and orthogonal to normal (cross product).
        // Additionally there is a small displacement in direction of the normal to elevate the triangles a little bit above the surface.
        // This way, the triangles stay visible even if the point size is increased.
        const uint32_t neigh_label = sv_adjacency_list[*itr_neighbor];
        const pcl::PointXYZRGBA& neigh_centroid = supervoxel_clusters[neigh_label]->centroid_;
        const pcl::Normal& neigh_normal = supervoxel_clusters[neigh_label]->normal_;


        // calculate difference vector
        pcl::PointXYZ diffVec;
        diffVec.x = neigh_centroid.x - sv_centroid.x;
        diffVec.y = neigh_centroid.y - sv_centroid.y;
        diffVec.z = neigh_centroid.z - sv_centroid.z;

        // Calculate cross product between difference vector and normal
        pcl::PointXYZ crossP;
        crossP.x = diffVec.y * neigh_normal.normal_z - diffVec.z * neigh_normal.normal_y;
        crossP.y = diffVec.z * neigh_normal.normal_x - diffVec.x * neigh_normal.normal_z;
        crossP.z = diffVec.x * neigh_normal.normal_y - diffVec.y * neigh_normal.normal_x;
        float crossPnorm = std::sqrt( crossP.x * crossP.x + crossP.y * crossP.y + crossP.z * crossP.z);
        crossP.x /= crossPnorm;
        crossP.y /= crossPnorm;
        crossP.z /= crossPnorm;

        // Third vertex of first triangle
        pcl::PointXYZRGBA vertex1;
        vertex1.x = neigh_centroid.x + 0.001 * crossP.x + 0.001 * neigh_normal.normal_x;
        vertex1.y = neigh_centroid.y + 0.001 * crossP.y + 0.001 * neigh_normal.normal_y;
        vertex1.z = neigh_centroid.z + 0.001 * crossP.z + 0.001 * neigh_normal.normal_z;
        vertex1.rgba = edge_color;

        mesh_vertices->push_back(vertex1);

        // Third vertex of second triangle
        pcl::PointXYZRGBA vertex2;
        vertex2.x = neigh_centroid.x + 0.001 * neigh_normal.normal_x + 0.001 * neigh_normal.normal_x;
        vertex2.y = neigh_centroid.y + 0.001 * neigh_normal.normal_y + 0.001 * neigh_normal.normal_y;
        vertex2.z = neigh_centroid.z + 0.001 * neigh_normal.normal_z + 0.001 * neigh_normal.normal_z;
        vertex2.rgba = edge_color;

        mesh_vertices->push_back(vertex2);

        // First triangle to draw
        pcl::Vertices triangle1;
        triangle1.vertices.push_back(mesh_vertices->size()-4); // This refers to vert_curr
        triangle1.vertices.push_back(mesh_vertices->size()-3); // This refers to vert_neigh
        triangle1.vertices.push_back(mesh_vertices->size()-2); // This refers to vertex1

        // Second triangle to draw
        pcl::Vertices triangle2;
        triangle2.vertices.push_back(mesh_vertices->size()-4); // This refers to vert_curr
        triangle2.vertices.push_back(mesh_vertices->size()-3); // This refers to vert_neigh
        triangle2.vertices.push_back(mesh_vertices->size()-1); // This refers to vertex2

        // Add the triangles to the triangle list
        mesh_triangles.push_back(triangle1);
        mesh_triangles.push_back(triangle2);
      }
    }
    /// END: Calculate visualization of adjacency graph


    /// Configure Visualizer
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->registerKeyboardCallback(keyboard_callback, 0);

    viewer->addPointCloud(input_CloudPtr, "maincloud");
    //     // Add Coordinate System
    //     viewer->addCoordinateSystem();
    //     pcl::PointXYZ origin;
    //     origin.x = 0;
    //     origin.y = 0;
    //     origin.z = 0;
    //     viewer->addSphere<pcl::PointXYZ>(origin,0.05);


    /// Visualization Loop
    std::cout << "Loading viewer..." << std::endl;
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);

      /// Show Segmentation or Supervoxels
      viewer->updatePointCloud( (show_supervoxels)?sv_colored_cloud:input_CloudPtr, "maincloud");


      /// Show Normals
      if(show_normals)
      {
        if( not normals_added )
        {
          viewer->addPointCloudNormals<pcl::PointNormal> (sv_centroid_normal_cloud, 1, 0.015, "supervoxels");
          normals_added = true;
        }

      }
      else
      {
        viewer->removePointCloud("supervoxels");
        normals_added = false;
      }

      /// Show Adjacency
      if(show_adjacency)
      {
        if( !viewer->updatePolygonMesh<pcl::PointXYZRGBA>(mesh_vertices, mesh_triangles, "adjacency_graph") )
          viewer->addPolygonMesh<pcl::PointXYZRGBA>(mesh_vertices, mesh_triangles, "adjacency_graph");

        // viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActors()->GetLastActor()->GetProperty()->SetInterpolationToFlat();
      }
      else
      {
        viewer->removePolygonMesh("adjacency_graph");
      }


      if (show_help)
      {
        viewer->removeShape ("help_text");
        printText (viewer);
      }
      else
      {
        removeText (viewer);
        if (!viewer->updateText("Press h to show help", 5, 10, 12, 1.0, 1.0, 1.0,"help_text") )
          viewer->addText("Press h to show help", 5, 10, 12, 1.0, 1.0, 1.0,"help_text");
      }

      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

  } /// END if(show_visualization)


  return (0);

} /// END main



/// ******  Definitions ****** ///


void
drawColors(std::vector< uint32_t >& label_colors, uint nr_labels)
{
  //   /// Create colors for coloring segments
  const uint32_t max_label = 1.1*nr_labels; //10% error margin..
  label_colors.clear();
  label_colors.reserve (max_label);
  //NOTE First label pushed back can be used for errors
  label_colors.push_back (static_cast<uint32_t>(0) << 16 | static_cast<uint32_t>(0) << 8 | static_cast<uint32_t>(0));

  srand (static_cast<unsigned int> (time (0)));
  for (size_t i_label = 0; i_label < max_label; i_label++)
  {
      uint8_t r = static_cast<uint8_t>( (rand () % 256));
      uint8_t g = static_cast<uint8_t>( (rand () % 256));
      uint8_t b = static_cast<uint8_t>( (rand () % 256));
      label_colors.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
  }
}


void
rgbd2pointcloud(std::string rgb_path, std::string depth_path, pcl::PointCloud<PointT>::Ptr output_cloud)
{
  //Read the images
  vtkSmartPointer<vtkImageReader2Factory> reader_factory = vtkSmartPointer<vtkImageReader2Factory>::New ();
  vtkImageReader2* rgb_reader = reader_factory->CreateImageReader2 (rgb_path.c_str ());
  //qDebug () << "RGB File="<< QString::fromStdString(rgb_path);
  if ( ! rgb_reader->CanReadFile (rgb_path.c_str ()))
  {
    std::cout << "Cannot read rgb image file!";
    exit(EXIT_FAILURE);
  }
  rgb_reader->SetFileName (rgb_path.c_str ());
  rgb_reader->Update ();
  //qDebug () << "Depth File="<<QString::fromStdString(depth_path);
  vtkImageReader2* depth_reader = reader_factory->CreateImageReader2 (depth_path.c_str ());
  if ( ! depth_reader->CanReadFile (depth_path.c_str ()))
  {
    std::cout << "Cannot read depth image file!";
    exit(EXIT_FAILURE);
  }
  depth_reader->SetFileName (depth_path.c_str ());
  depth_reader->Update ();

  vtkSmartPointer<vtkImageFlip> flipXFilter = vtkSmartPointer<vtkImageFlip>::New();
  flipXFilter->SetFilteredAxis(0); // flip x axis
  flipXFilter->SetInputConnection(rgb_reader->GetOutputPort());
  flipXFilter->Update();

  vtkSmartPointer<vtkImageFlip> flipXFilter2 = vtkSmartPointer<vtkImageFlip>::New();
  flipXFilter2->SetFilteredAxis(0); // flip x axis
  flipXFilter2->SetInputConnection(depth_reader->GetOutputPort());
  flipXFilter2->Update();

  vtkSmartPointer<vtkImageData> rgb_image = flipXFilter->GetOutput ();
  int *rgb_dims = rgb_image->GetDimensions ();
  vtkSmartPointer<vtkImageData> depth_image = flipXFilter2->GetOutput ();
  int *depth_dims = depth_image->GetDimensions ();

  if (rgb_dims[0] != depth_dims[0] || rgb_dims[1] != depth_dims[1])
  {
    std::cout << "Depth and RGB dimensions to not match!";
    std::cout << "RGB Image is of size "<<rgb_dims[0] << " by "<<rgb_dims[1];
    std::cout << "Depth Image is of size "<<depth_dims[0] << " by "<<depth_dims[1];
    exit(EXIT_FAILURE);
  }

  output_cloud->points.reserve (depth_dims[0] * depth_dims[1]);
  output_cloud->width = depth_dims[0];
  output_cloud->height = depth_dims[1];
  output_cloud->is_dense = false;


  // Fill in image data
  int centerX = static_cast<int>(output_cloud->width / 2.0);
  int centerY = static_cast<int>(output_cloud->height / 2.0);
  unsigned short* depth_pixel;
  unsigned char* color_pixel;
  float scale = 1.0f/1000.0f;
  float focal_length = 525.0f;
  float fl_const = 1.0f / focal_length;
  depth_pixel = static_cast<unsigned short*>(depth_image->GetScalarPointer (depth_dims[0]-1,depth_dims[1]-1,0));
  color_pixel = static_cast<unsigned char*> (rgb_image->GetScalarPointer (depth_dims[0]-1,depth_dims[1]-1,0));

  #pragma GCC diagnostic push // remember warning settings
  #pragma GCC diagnostic ignored "-Wsign-compare"
  for (int y=0; y<output_cloud->height; ++y)
  {
    for (int x=0; x<output_cloud->width; ++x, --depth_pixel, color_pixel-=3)
    {
      PointT new_point;
      //  uint8_t* p_i = &(cloud_blob->data[y * cloud_blob->row_step + x * cloud_blob->point_step]);
      float depth = static_cast<float>(*depth_pixel) * scale;
      if (depth == 0.0f)
      {
        new_point.x = new_point.y = new_point.z = std::numeric_limits<float>::quiet_NaN ();
      }
      else
      {
        new_point.x = (static_cast<float>(x - centerX)) * depth * fl_const;
        new_point.y = (static_cast<float>(centerY - y)) * depth * fl_const; // vtk seems to start at the bottom left image corner
        new_point.z = depth;
      }

      uint32_t rgb = static_cast<uint32_t>(color_pixel[0]) << 16 |  static_cast<uint32_t>(color_pixel[1]) << 8 |  static_cast<uint32_t>(color_pixel[2]);
      new_point.rgb = *reinterpret_cast<float*> (&rgb);
      output_cloud->points.push_back (new_point);

    }
  }
  #pragma GCC diagnostic pop // restore warning setttings
}



void printText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  std::string on_str = "[ON]";
  std::string off_str = "[OFF]";
  if (!viewer->updateText ("Press (1-n) to show different elements (h) to disable this", 5, 72, 12, 1.0, 1.0, 1.0,"hud_text"))
    viewer->addText ("Press (1-n) to show different elements", 5, 72, 12, 1.0, 1.0, 1.0,"hud_text");

  std::string temp = "(1) Supervoxel Normals currently " + ((show_normals)?on_str:off_str);
  if (!viewer->updateText (temp, 5, 60, 10, 1.0, 1.0, 1.0, "normals_text"))
    viewer->addText (temp, 5, 60, 10, 1.0, 1.0, 1.0, "normals_text");

  temp = "(2) Adjacency Graph currently "+ ((show_adjacency)?on_str:off_str) + "\n      White: convex; Red: concave";
  if (!viewer->updateText (temp, 5, 38, 10, 1.0, 1.0, 1.0, "graph_text") )
    viewer->addText (temp, 5, 38, 10, 1.0, 1.0, 1.0, "graph_text");

  temp = "(3) Press to show "+ ((show_supervoxels)?std::string("SEGMENTATION"):std::string("SUPERVOXELS"));
  if (!viewer->updateText (temp, 5, 26, 10, 1.0, 1.0, 1.0, "supervoxel_text") )
    viewer->addText (temp, 5, 26, 10, 1.0, 1.0, 1.0, "supervoxel_text");
}

void removeText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  viewer->removeShape ("hud_text");
  viewer->removeShape ("normals_text");
  viewer->removeShape ("graph_text");
  viewer->removeShape ("supervoxel_text");
}



