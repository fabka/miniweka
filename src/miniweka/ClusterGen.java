/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package miniweka;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author estudiante
 */
public class ClusterGen {
    
    Instances data;
    Instances instancesFilter;
    String path;
    
    /**
     * 
     * @param path
     * @throws FileNotFoundException
     * @throws IOException 
     */
    public ClusterGen(String path) throws FileNotFoundException, IOException {
        try (BufferedReader reader = new BufferedReader(
                new FileReader(path))) {
            this.path = path;
            data = new Instances(reader);
        }
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);
    }
    
    /**
     * 
     * @param attribute 
     */
    public void imprimirAtributo(int attribute) {
        for(Double d: data.attributeToDoubleArray(0)){
            System.out.println(d);
        }
    }
    
    /**
     *
     * @param df
     * @param numCluster
     * @param seed
     * @param maxIterations
     * @param replaceMissingValues
     * @param preserveInstancesOrder
     * @param atributos
     * @throws Exception
     */
    public void kmeans (DistanceFunction df, int numCluster, int seed, int maxIterations,
            boolean replaceMissingValues, boolean preserveInstancesOrder, Vector<Integer> atributos) throws Exception{
        
        Instances anonimized = new Instances(data);
        
        //Verificar atributos
        for( Integer n: atributos ){
            if(data.attribute(n-1).isDate())
                throw new Exception("Cannot handle date attributes!");
            else if(data.attribute(n-1).isString())
                throw new Exception("Cannot handle string attributes!");
        }
        
        Remove remove = new Remove();
        remove.setAttributeIndices(getAttributesIndex(atributos));
        remove.setInvertSelection(true);
        remove.setInputFormat(data);
        Instances dataCopy = Filter.useFilter(data, remove);
        
        SimpleKMeans kmeans = new SimpleKMeans();
                
        kmeans.setNumClusters(numCluster);
        kmeans.setMaxIterations(maxIterations);
        kmeans.setSeed(seed);
        kmeans.setDisplayStdDevs(false);
        kmeans.setDistanceFunction(df);
        kmeans.setDontReplaceMissingValues(replaceMissingValues);
        kmeans.setPreserveInstancesOrder(preserveInstancesOrder);
        kmeans.buildClusterer(dataCopy);
        
        int i;
        int[] assignments = kmeans.getAssignments();
        Instances centroids = kmeans.getClusterCentroids();
        for( i=0; i<dataCopy.numInstances(); i++ ){
            int nInstancia = assignments[i];
            Instance instanciaCentroide = centroids.get(nInstancia);
            for(int j=0; j<dataCopy.numAttributes(); j++){
                Double valor = instanciaCentroide.value(j);
                dataCopy.instance(i).setValue(j, valor);
            }
        }
        for( i=0; i<atributos.size(); i++ ){
            for(int j = 0; j<anonimized.numInstances(); j++){
                int atributo_copia = atributos.get(i);
                int atributo = i;
                double valor = dataCopy.instance(j).value(atributo);
                anonimized.instance(j).setValue(atributo, valor);
            }
        }
        
        exportARFF(anonimized);
        
        /*for( Instance instance: dataCopy ){
            System.out.println(instance);
        }
                
        for(int clusterNum : assignments) {
            System.out.printf("Instance %d -> Cluster %d \n", i, clusterNum);
            i++;
        }
        
        for (i = 0; i < centroids.numInstances(); i++) { 
          System.out.println( "Centroid " + i + ": " + centroids.instance(i)); 
        }*/
    }
    
    private String getAttributesIndex(Vector<Integer> atributos) {
        String indices="";
        int i;
        for(i=0; i<atributos.size()-1; i++){
            indices += atributos.get(i)+",";
        }
        indices += atributos.get(i);
        return indices;
    }

    private void exportARFF(Instances instances) {
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(instances);
            String[] splitted = path.split(".arff");
            saver.setFile(new File(splitted[0]+"_anonimizado.arff"));
            saver.writeBatch();
        } catch (IOException ex) {
            Logger.getLogger(ClusterGen.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
