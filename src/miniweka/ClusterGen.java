/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package miniweka;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

/**
 *
 * @author estudiante
 */
public class ClusterGen {
    
    String ruta;

    public ClusterGen() {
    }
        
    public double obtenerCentroide( int numberOfClusters, Instances instances) throws Exception{
        SimpleKMeans kmeans = new SimpleKMeans();

        kmeans.setSeed(10);

        // This is the important parameter to set
        kmeans.setPreserveInstancesOrder(true);
        kmeans.setNumClusters(numberOfClusters);
        
        kmeans.buildClusterer(instances);

        // This array returns the cluster number (starting with 0) for each instance
        // The array has as many elements as the number of instances
        int[] assignments = kmeans.getAssignments();

        int i=0;
        for(int clusterNum : assignments) {
            System.out.printf("Instance %d -> Cluster %d", i, clusterNum);
            i++;
        }
        
        return 0;
    }
    
}
