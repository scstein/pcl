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


#ifndef PCL_SEGMENTATION_LCCP_IMPL_H_
#define PCL_SEGMENTATION_LCCP_IMPL_H_


#include <pcl/segmentation/lccp_segmentation.h>


template <typename PointT>
pcl::LCCPSegmentation<PointT>::LCCPSegmentation ():
  concavity_tolerance_threshold_(10),
  grouping_data_valid_(false),
  use_smoothness_check_(false),
  smoothness_threshold_(0.1),
  seed_resolution_(0),
  voxel_resolution_(0),
  k_factor_(0)
{

};


template <typename PointT>
pcl::LCCPSegmentation<PointT>::~LCCPSegmentation()
{
}



template <typename PointT> float
pcl::LCCPSegmentation<PointT>::vecNorm(const pcl::Normal& x) const
{
  float norm;
  norm = std::sqrt(x.normal_x*x.normal_x +
                    x.normal_y*x.normal_y +
                    x.normal_z*x.normal_z);
  return (norm);
}

template <typename PointT> float
pcl::LCCPSegmentation<PointT>::vecNorm(const pcl::PointXYZ& x) const
{
  float norm;
  norm = std::sqrt(x.x*x.x +  x.y*x.y +  x.z*x.z);
  return (norm);
}



template <typename PointT> float
pcl::LCCPSegmentation<PointT>::dotProduct(const pcl::Normal& x, const pcl::Normal& y) const
{
  float dotP = 0;
  dotP += (x.normal_x * y.normal_x);
  dotP += (x.normal_y * y.normal_y);
  dotP += (x.normal_z * y.normal_z);
  return (dotP);
}

template <typename PointT> float
pcl::LCCPSegmentation<PointT>::dotProduct(const pcl::PointXYZ& x, const pcl::PointXYZ& y) const
{
  float dotP = 0;
  dotP += (x.x * y.x);
  dotP += (x.y * y.y);
  dotP += (x.z * y.z);
  return (dotP);
}


template <typename PointT> float
pcl::LCCPSegmentation<PointT>::dotProduct(const pcl::PointXYZ& x, const pcl::Normal& y) const
{
  float dotP = 0;
  dotP += (x.x * y.normal_x);
  dotP += (x.y * y.normal_y);
  dotP += (x.z * y.normal_z);
  return (dotP);
}

template <typename PointT> float
pcl::LCCPSegmentation<PointT>::dotProduct(const pcl::Normal& x, const pcl::PointXYZ& y) const
{
  float dotP = 0;
  dotP += (x.normal_x * y.x);
  dotP += (x.normal_y * y.y);
  dotP += (x.normal_z * y.z);
  return (dotP);
}




template <typename PointT> pcl::Normal
pcl::LCCPSegmentation<PointT>::crossP(const pcl::Normal& x, const pcl::Normal& y) const
{
  pcl::Normal z;
  z.normal_x = x.normal_y*y.normal_z - x.normal_z*y.normal_y;
  z.normal_y = x.normal_z*y.normal_x - x.normal_x*y.normal_z;
  z.normal_z = x.normal_x*y.normal_y - x.normal_y*y.normal_x;
  return (z);
}



template <typename PointT> void
pcl::LCCPSegmentation<PointT>::normalizeVec(pcl::Normal& x) const
{
  float norm;
  norm = vecNorm(x);
  x.normal_x /=norm;
  x.normal_y /=norm;
  x.normal_z /=norm;
}

template <typename PointT> void
pcl::LCCPSegmentation<PointT>::normalizeVec(pcl::PointXYZ& x) const
{
  float norm;
  norm = vecNorm(x);
  x.x /=norm;
  x.y /=norm;
  x.z /=norm;
}



template <typename PointT> void
pcl::LCCPSegmentation<PointT>::reset()
{
  sv_adjacency_list_.clear();
  processed_.clear();
  svLabel_supervoxel_map_.clear();
  svLabel_segLabel_map_.clear();
  segLabel_svlist_map_.clear();
  segLabel_neighborSet_map_.clear();
  grouping_data_valid_ = false;
}



template <typename PointT> void
pcl::LCCPSegmentation<PointT>::getSegmentSupervoxelMap(std::map< boost::uint32_t, std::vector< boost::uint32_t > >& segment_supervoxel_map_arg) const
{
  if(grouping_data_valid_)
  {
    segment_supervoxel_map_arg = segLabel_svlist_map_;
  }
  else
    PCL_ERROR("[pcl::LCCPSegmentation::getSegmentMap] ERROR: No valid data from grouping.\n");
}



template <typename PointT> void
pcl::LCCPSegmentation<PointT>::getSegmentAdjacencyMap(std::map< boost::uint32_t, std::set< boost::uint32_t > >& segment_adjacency_map_arg)
{
  if(grouping_data_valid_)
  {
    if(segLabel_neighborSet_map_.empty())
      computeSegmentAdjacency();

    segment_adjacency_map_arg = segLabel_neighborSet_map_;
  }
  else
    PCL_ERROR("[pcl::LCCPSegmentation::getSegmentAdjacencyMap] ERROR: No valid data from grouping.");
}



template <typename PointT> void
pcl::LCCPSegmentation<PointT>::relabelCloud(pcl::PointCloud< pcl::PointXYZL >::Ptr labeled_cloud_arg)
{
  if(grouping_data_valid_)
  {
    /// Relabel all Points in cloud with new labels
    typename pcl::PointCloud <pcl::PointXYZL>::iterator i_voxel = labeled_cloud_arg->begin ();
    for(; i_voxel != labeled_cloud_arg->end() ; ++i_voxel)
    {
      i_voxel->label = svLabel_segLabel_map_[ i_voxel->label ] ;
    }
  }
  else
    PCL_ERROR("[pcl::LCCPSegmentation::relabelCloud] ERROR: No valid data from grouping.");
}



template <typename PointT> void
pcl::LCCPSegmentation<PointT>::computeSegmentAdjacency()
{
  if(grouping_data_valid_)
  {
    segLabel_neighborSet_map_.clear();

    //The vertices in the supervoxel adjacency list are the supervoxel centroids
    std::pair<VertexIterator, VertexIterator> vertex_iterator_range;
    vertex_iterator_range = boost::vertices(sv_adjacency_list_);

    uint32_t current_segLabel;
    uint32_t neigh_segLabel;

    /// For every Supervoxel..
    for (VertexIterator sv_itr=vertex_iterator_range.first ; sv_itr != vertex_iterator_range.second; ++sv_itr) // For all SuperVoxels
    {
      const uint32_t SV_LABEL = sv_adjacency_list_[*sv_itr];
      current_segLabel = svLabel_segLabel_map_[SV_LABEL];

      /// ..look at all neighbors and insert their labels into the neighbor set
      std::pair<AdjacencyIterator, AdjacencyIterator> neighbors = boost::adjacent_vertices (*sv_itr, sv_adjacency_list_ );
      for(AdjacencyIterator itr_neighbor = neighbors.first; itr_neighbor != neighbors.second; ++itr_neighbor)
      {
        const uint32_t NEIGH_LABEL = sv_adjacency_list_[*itr_neighbor];
        neigh_segLabel = svLabel_segLabel_map_[NEIGH_LABEL];

        if( current_segLabel != neigh_segLabel )
        {
          segLabel_neighborSet_map_[current_segLabel].insert(neigh_segLabel);
        }
      }
    }
  }
  else
    PCL_ERROR("[pcl::LCCPSegmentation::computeSegmentAdjacency] ERROR: No valid data from grouping.");
}


template <typename PointT> void
pcl::LCCPSegmentation<PointT>::removeNoise(uint32_t filter_size)
{
  if(filter_size==0) return;

  if(grouping_data_valid_)
  {
    if(segLabel_neighborSet_map_.empty())
        computeSegmentAdjacency();


    std::set<uint32_t> filteredSegLabels;

    uint32_t largest_neigh_size = 0;
    uint32_t largest_neigh_segLabel = 0;
    uint32_t current_segLabel;

    std::pair<VertexIterator, VertexIterator> vertex_iterator_range;
    vertex_iterator_range = boost::vertices(sv_adjacency_list_);

    bool continueFiltering = true;


    while(continueFiltering)
    {
      continueFiltering = false;
      uint nr_filtered = 0;

      /// Iterate through all supervoxels, check if they are in a "small" segment -> change label to largest neighborID
      for (VertexIterator sv_itr=vertex_iterator_range.first ; sv_itr != vertex_iterator_range.second; ++sv_itr) // For all SuperVoxels
      {
        const uint32_t SV_LABEL = sv_adjacency_list_[*sv_itr];
        current_segLabel = svLabel_segLabel_map_[SV_LABEL];
        largest_neigh_segLabel = current_segLabel;
        largest_neigh_size = segLabel_svlist_map_[current_segLabel].size();

        const uint32_t NR_NEIGHBORS = segLabel_neighborSet_map_[current_segLabel].size();
        if(NR_NEIGHBORS == 0) continue;

        if(segLabel_svlist_map_[current_segLabel].size() <= filter_size)
        {
          continueFiltering = true;
          nr_filtered++;

          // Find largest neighbor
          std::set< uint32_t >::const_iterator it_neighbors = segLabel_neighborSet_map_[current_segLabel].begin();
          for(; it_neighbors != segLabel_neighborSet_map_[current_segLabel].end(); ++it_neighbors )
          {
            if(segLabel_svlist_map_[*it_neighbors].size() >= largest_neigh_size)
            {
              largest_neigh_segLabel = *it_neighbors;
              largest_neigh_size = segLabel_svlist_map_[*it_neighbors].size();
            }
          }

          // Add to largest neighbor
          if(largest_neigh_segLabel != current_segLabel)
          {
            if( filteredSegLabels.count(largest_neigh_segLabel) > 0) continue; // If neighbor was already assigned to someone else

            svLabel_segLabel_map_[SV_LABEL] = largest_neigh_segLabel;
            filteredSegLabels.insert(current_segLabel);

            // Assign SuperVoxel labels of filtered segment to new owner
            std::vector< uint32_t >::iterator it_svID = segLabel_svlist_map_[current_segLabel].begin();
            it_svID = segLabel_svlist_map_[current_segLabel].begin();
            for(; it_svID != segLabel_svlist_map_[current_segLabel].end(); ++it_svID)
            {
              segLabel_svlist_map_[largest_neigh_segLabel].push_back(*it_svID);
            }

          }
        }
      }


      /// Erase filtered Segments from segment map
      std::set<uint32_t>::iterator it_filteredID = filteredSegLabels.begin();
      for(; it_filteredID != filteredSegLabels.end(); ++it_filteredID)
      {
        segLabel_svlist_map_.erase(*it_filteredID);
      }

      ///After filtered Segments are deleted, compute completely new adjacency map
      // NOTE Recomputing the adjacency of every segment in every iteration is an easy but inefficient solution.
      // Because the number of segments in an average scene is usually well below 1000, the time spend for noise filtering is still neglible in most cases
      computeSegmentAdjacency();

      // std::cout << "Filtered " << nr_filtered << " segments." << std::endl;
    } // End while(Filtering)

  }
  else
    PCL_ERROR("[pcl::LCCPSegmentation::removeNoise] ERROR: No valid data from grouping.");
}




template <typename PointT> void
pcl::LCCPSegmentation<PointT>::prepareSegmentation(const std::map< uint32_t, typename pcl::Supervoxel< PointT >::Ptr >& SUPERVOXEL_CLUSTERS_ARG, const std::multimap< boost::uint32_t, boost::uint32_t >& LABEL_ADJACENCY_ARG)
{
  // Clear internal data
  reset();

  // Copy map with supervoxel pointers
  svLabel_supervoxel_map_ = SUPERVOXEL_CLUSTERS_ARG;

  ///    Build a boost adjacency list from the adjacency multimap
  std::map<uint32_t, VertexID> label_ID_map;

  // Add all supervoxel labels as vertices
  for(typename std::map <uint32_t, typename pcl::Supervoxel<PointT>::Ptr >::iterator it_svlabel = svLabel_supervoxel_map_.begin(); it_svlabel != svLabel_supervoxel_map_.end(); ++it_svlabel)
  {
    const uint32_t SV_LABEL = it_svlabel->first;
    VertexID node_id = boost::add_vertex (sv_adjacency_list_);
    sv_adjacency_list_[node_id] = SV_LABEL;
    label_ID_map[SV_LABEL] = node_id;
  }

  // Add all edges
  for(std::multimap<uint32_t, uint32_t>::const_iterator it_sv_neighbors = LABEL_ADJACENCY_ARG.begin(); it_sv_neighbors != LABEL_ADJACENCY_ARG.end(); ++it_sv_neighbors)
  {
    const uint32_t SV_LABEL = it_sv_neighbors->first;
    const uint32_t NEIGHBOR_LABEL = it_sv_neighbors->second;

    VertexID u = label_ID_map[SV_LABEL];
    VertexID v = label_ID_map[NEIGHBOR_LABEL];

    bool edge_added;
    EdgeID edge;

    boost::tie (edge, edge_added) = boost::add_edge (u,v,sv_adjacency_list_);
  }


  // Initialization
  for(typename std::map <uint32_t, typename pcl::Supervoxel<PointT>::Ptr >::iterator it_svlabel = svLabel_supervoxel_map_.begin(); it_svlabel != svLabel_supervoxel_map_.end(); ++it_svlabel)
  {
    const uint32_t SV_LABEL = it_svlabel->first;
    processed_[SV_LABEL] = false;
    svLabel_segLabel_map_[SV_LABEL] = 0;
  }
}


template <typename PointT> void
pcl::LCCPSegmentation<PointT>::segment(std::map< uint32_t, typename pcl::Supervoxel< PointT >::Ptr >& supervoxel_clusters_arg, std::multimap< boost::uint32_t, boost::uint32_t >& label_adjacency_arg)
{
  /// Initialization
  prepareSegmentation(supervoxel_clusters_arg, label_adjacency_arg); // after this, sv_adjacency_list_ can be used to access adjacency list

  /// Calculate for every Edge if the connection is convex or invalid
  /// This effectively performs the segmentation.
  calculateConvexConnections( sv_adjacency_list_ );

  /// Correct edge relations using extended convexity definition if k>0
  applyKconvexity( k_factor_ );


  /// Perform depth search on the graph and recursively group all supervoxels with convex connections
  //The vertices in the supervoxel adjacency list are the supervoxel centroids
  std::pair<VertexIterator, VertexIterator> vertex_iterator_range;
  vertex_iterator_range = boost::vertices(sv_adjacency_list_);


  // Note: *sv_itr is of type " boost::graph_traits<VoxelAdjacencyList>::vertex_descriptor " which it nothing but a typedef of size_t..
  unsigned int segment_label = 1; // This starts at 1, because 0 is reserved for errors
  for (VertexIterator sv_itr=vertex_iterator_range.first ; sv_itr != vertex_iterator_range.second; ++sv_itr) // For all SuperVoxels
  {
    const VertexID SV_VERTEX_ID = *sv_itr;
    const uint32_t SV_LABEL = sv_adjacency_list_[SV_VERTEX_ID];
    if(!processed_[SV_LABEL])
    {
      // Add neighbors (and their neighbors etc.) to group if similarity constraint is met
      recursiveGrouping(SV_VERTEX_ID, segment_label);
      ++segment_label; // After recursive grouping ended (no more neighbors to consider) -> go to next group
    }

  }

  grouping_data_valid_ = true;
}


template <typename PointT> void
pcl::LCCPSegmentation<PointT>::recursiveGrouping(VertexID const &QUERY_POINT_ID, unsigned int const SEGMENT_LABEL)
{
  const uint32_t sv_label = sv_adjacency_list_[QUERY_POINT_ID];

  processed_[sv_label] = true;

  // The next two lines add the supervoxel to the segment
  svLabel_segLabel_map_[sv_label] = SEGMENT_LABEL;
  segLabel_svlist_map_[SEGMENT_LABEL].push_back(sv_label);

  /// Iterate through all neighbors of this supervoxel and check wether they should be merged with the current SuperVoxel
  std::pair<OutEdgeIterator, OutEdgeIterator> outEdge_iterator_range;
  outEdge_iterator_range = boost::out_edges(QUERY_POINT_ID,sv_adjacency_list_); // adjacent vertices to node (*itr) in graph sv_adjacency_list_
  for (OutEdgeIterator outEdge_itr=outEdge_iterator_range.first ; outEdge_itr != outEdge_iterator_range.second; ++outEdge_itr)
  {
    const VertexID neighborID = boost::target(*outEdge_itr, sv_adjacency_list_);
    const uint32_t neighbor_label = sv_adjacency_list_[neighborID];

    if(!processed_[neighbor_label]) // If neighbor was not already processed
    {
      if( sv_adjacency_list_[*outEdge_itr].isConvex )
      {
        recursiveGrouping(neighborID, SEGMENT_LABEL);
      }
    }

  } // End neighbor loop
}





template <typename PointT> void
pcl::LCCPSegmentation<PointT>::applyKconvexity(uint k)
{
  if(k==0) return;

  bool isConvex;
  uint kcount = 0;

  EdgeIterator edge_itr, edge_itr_end, next_edge;
  boost::tie(edge_itr, edge_itr_end) = boost::edges(sv_adjacency_list_);

  std::pair<OutEdgeIterator, OutEdgeIterator> source_neighbors_range;
  std::pair<OutEdgeIterator, OutEdgeIterator> target_neighbors_range;


  // Check all edges in the graph for k-convexity
  for(next_edge = edge_itr; edge_itr != edge_itr_end; edge_itr = next_edge)
  {
    next_edge++; // next_edge iterator is neccessary, because removing an edge invalidates the iterator to the current edge

    isConvex = sv_adjacency_list_[*edge_itr].isConvex;

    if( isConvex ) // If edge is (0-)convex
    {
      kcount = 0;

      const VertexID source = boost::source(*edge_itr, sv_adjacency_list_);
      const VertexID target = boost::target(*edge_itr, sv_adjacency_list_);

      source_neighbors_range = boost::out_edges(source,sv_adjacency_list_);
      target_neighbors_range = boost::out_edges(target,sv_adjacency_list_);

      // Find common neighbors, check their connection
      for (OutEdgeIterator source_neighbors_itr=source_neighbors_range.first ; source_neighbors_itr != source_neighbors_range.second; ++source_neighbors_itr) // For all SuperVoxels
      {
        VertexID source_neighborID = boost::target(*source_neighbors_itr, sv_adjacency_list_);

        for (OutEdgeIterator target_neighbors_itr=target_neighbors_range.first ; target_neighbors_itr != target_neighbors_range.second; ++target_neighbors_itr) // For all SuperVoxels
        {
          VertexID target_neighborID = boost::target(*target_neighbors_itr, sv_adjacency_list_);
          if(source_neighborID == target_neighborID) // Common neighbor
          {
            EdgeID src_edge = boost::edge(source, source_neighborID, sv_adjacency_list_).first;
            EdgeID tar_edge = boost::edge(target, source_neighborID, sv_adjacency_list_).first;

            bool src_isConvex = (sv_adjacency_list_)[src_edge].isConvex;
            bool tar_isConvex = (sv_adjacency_list_)[tar_edge].isConvex;

            if(src_isConvex and tar_isConvex)
              ++kcount;


            break;
          }
        }

        if(kcount >= k) // Connection is k-convex, stop search
          break;
      }

      /// Check k convexity
      if(kcount < k)
        (sv_adjacency_list_)[*edge_itr].isConvex = false;
    }

  }


}




template <typename PointT> void
pcl::LCCPSegmentation<PointT>::calculateConvexConnections(SupervoxelAdjacencyList& adjacency_list)
{
  bool isConvex;

  EdgeIterator edge_itr, edge_itr_end, next_edge;
  boost::tie(edge_itr, edge_itr_end) = boost::edges(adjacency_list);

  for(next_edge = edge_itr; edge_itr != edge_itr_end; edge_itr = next_edge)
  {
    next_edge++; // next_edge iterator is neccessary, because removing an edge invalidates the iterator to the current edge

    uint32_t source_sv_label = adjacency_list[boost::source(*edge_itr, adjacency_list)];
    uint32_t target_sv_label = adjacency_list[boost::target(*edge_itr, adjacency_list)];

    isConvex = connIsConvex(source_sv_label, target_sv_label);
    adjacency_list[*edge_itr].isConvex = isConvex;

//     if(!isConvex) boost::remove_edge(*edge_itr, *adj_list_ptr_); // remove concave connections, //FIXME If this is set, smoothing won't work anymore!
  }

}


template <typename PointT> bool
pcl::LCCPSegmentation<PointT>::connIsConvex(uint32_t source_label, uint32_t target_label)
{

  typename pcl::Supervoxel<PointT>::Ptr& sv_source = svLabel_supervoxel_map_[source_label];
  typename pcl::Supervoxel<PointT>::Ptr& sv_target = svLabel_supervoxel_map_[target_label];

  const pcl::PointXYZRGBA& SOURCE_CENTROID = sv_source->centroid_;
  const pcl::PointXYZRGBA& TARGET_CENTROID = sv_target->centroid_;

  const pcl::Normal& SOURCE_NORMAL = sv_source->normal_;
  const pcl::Normal& TARGET_NORMAL = sv_target->normal_;

  //NOTE For angles below 0 nothing will be merged
  if(concavity_tolerance_threshold_ < 0)
  {
    return (false);
  }

  bool isConvex = true;
  bool isSmooth = true;
  float normal_angle = std::acos(dotProduct(SOURCE_NORMAL, TARGET_NORMAL))*180./M_PI;
  //   float curvature_difference = std::fabs(SOURCE_NORMAL.curvature - TARGET_NORMAL.curvature);


  ///  Geometric comparisons
  pcl::PointXYZ vec_t_to_s, vec_s_to_t;
  pcl::PointXYZ unitvec_t_to_s, unitvec_s_to_t;
  vec_t_to_s.x = SOURCE_CENTROID.x - TARGET_CENTROID.x;
  vec_t_to_s.y = SOURCE_CENTROID.y - TARGET_CENTROID.y;
  vec_t_to_s.z = SOURCE_CENTROID.z - TARGET_CENTROID.z;
  unitvec_t_to_s = vec_t_to_s;
  normalizeVec(unitvec_t_to_s);

  vec_s_to_t.x = -vec_t_to_s.x;
  vec_s_to_t.y = -vec_t_to_s.y;
  vec_s_to_t.z = -vec_t_to_s.z;

  unitvec_s_to_t.x = -unitvec_t_to_s.x;
  unitvec_s_to_t.y = -unitvec_t_to_s.y;
  unitvec_s_to_t.z = -unitvec_t_to_s.z;

  float dotP_source, dotP_target;
  // float dotP_source_inv, dotPcheck;

  // vec_t_to_s is the reference direction for angle measurements
  dotP_source  = dotProduct(unitvec_t_to_s, SOURCE_NORMAL);
  dotP_target = dotProduct(unitvec_t_to_s, TARGET_NORMAL);
//   dotP_source_inv = dotProduct(unitvec_s_to_t, SOURCE_NORMAL);
//   dotPcheck = dotProduct(unitvec_s_to_t, TARGET_NORMAL);

  pcl::Normal ncross = crossP(SOURCE_NORMAL, TARGET_NORMAL);


  /// Smoothness Check: Check if there is a step between adjacent patches
  if(use_smoothness_check_)
  {
    float expectedDistance = vecNorm(ncross)*seed_resolution_;
    float dotP1 = dotProduct(vec_t_to_s, SOURCE_NORMAL);
    float dotP2 = dotProduct(vec_s_to_t, TARGET_NORMAL);
    float pointDist = (std::fabs(dotP1) < std::fabs(dotP2)) ? std::fabs(dotP1) : std::fabs(dotP2);
    const float DIST_SMOOTHING = smoothness_threshold_*voxel_resolution_; // This is a slacking variable especially important for patches with very similar normals

    if(pointDist > (expectedDistance + DIST_SMOOTHING))
    {
      isSmooth &= false;
    }
  }
  /// ----------------


  /// Sanity Criterion: Check if definition convexity/concavity makes sense for connection of given patches
  normalizeVec(ncross);

  float intersection_angle  = std::acos(dotProduct(ncross, unitvec_t_to_s))*180./M_PI;
  float min_intersect_angle = (intersection_angle < 90.) ? intersection_angle : 180.-intersection_angle;

  float intersect_thresh = 60.*1./(1. + exp(-0.25*(normal_angle-25.)) );
  if( min_intersect_angle < intersect_thresh)
  {
    // std::cout << "Concave/Convex not defined for given case!" << std::endl;
    isConvex &= false;
  }


  /// Convexity Criterion: Check if connection of patches is convex. If this is the case the two SuperVoxels should be merged.
  if( (std::acos(dotP_source) - std::acos(dotP_target)) <= 0)
  {
    isConvex &= true; // connection convex
  }
  else
  {
    isConvex &= (normal_angle < concavity_tolerance_threshold_); // concave connections will be accepted  if difference of normals is small
  }


  /** Alternative convexity criterion, also used in "Segmentation of 3D Lidar Data in non-flat Urban Environments
  using a Local Convexity Criterion" by Moosmann et al. */
  /*
    {
      float normal_noise_threshold = 5.; // Degrees allowed for noise

      isConvex&=     (dotP_source_inv < std::cos(M_PI/2. - concavity_tolerance_threshold_*M_PI/180.) \
                  and dotP_target < std::cos(M_PI/2. - concavity_tolerance_threshold_*M_PI/180.)) \
                  or (normal_angle < normal_noise_threshold);

    }
  */

  ///   FIXME This is a convexity criterion similar to Moosmanns, but where the threshold still corresponds to actually present geometric transition
  /*
    if( dotP_source_inv < 0  and dotP_target < 0)
    {
      isConvex &= true; // connection convex
    }
    else
    {
      isConvex &= (normal_angle < concavity_tolerance_threshold_); // concave connections will be accepted  if difference of normals is small
    }
  */


  return (isConvex and isSmooth);
}


#endif // PCL_SEGMENTATION_LCCP_IMPL_H_
