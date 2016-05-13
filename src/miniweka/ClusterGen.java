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
import java.util.HashMap;
import java.util.Map;
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
    String path;
    
    /**
     * Crea una serie de instancias weka para crear el cluster
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
    
    public ClusterGen( Instances instances ){
        this.data = instances;
    }
    
    /**
     * Imprime todos los valores del atributo attribute
     * 
     * @param attribute 
     */
    public void imprimirAtributo(int attribute) {
        for(Double d: data.attributeToDoubleArray(0)){
            System.out.println(d);
        }
    }
    
    /**
     * Genera el cluster, para ello necesita todas las variables que normalmente
     * se envían al programa de weka.
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
        
        int k = 3;
        boolean seCumple = verificarK(anonimized, k); 
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
    
    /**
     * De una lista de atributos a eliminar obtiene los indices y los coloca en
     * un formato entendible para la api de weka
     * 
     * @param atributos
     * @return 
     */
    private String getAttributesIndex(Vector<Integer> atributos) {
        String indices="";
        int i;
        for(i=0; i<atributos.size()-1; i++){
            indices += atributos.get(i)+",";
        }
        indices += atributos.get(i);
        return indices;
    }
    
    private Vector<Integer> getAttributesVector(String atributos) {
        String[] splitted = atributos.split(",");
        Vector<Integer> nAtributos =new Vector<>();
        for(String s: splitted){
            nAtributos.add(Integer.parseInt(s));
        }
        return nAtributos;
    }

    /**
     * Exporta una instancia a un formato arff para que no se pierda nungún dato.
     * 
     * @param instances 
     */
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

    public void anonimizar(Instances instances, Vector<Integer> atributos, int k){
        
        final int maxIteraciones = 500;
        int iteracion = 0;
        do{
            int atributo = mayorFrecuencia(atributos);

            if( esCompleto(atributo) ){
                iteracion++;//aplicar filtro 5 
            }else{
                iteracion++;//aplicar filtro 3
            }
        }while(true && iteracion < maxIteraciones); //Hacer mientras no se cumpla el k
        borrarTuplas(instances, k);
        exportARFF(instances);
    }
    
    private Instances borrarTuplas(Instances instances, int k){
        HashMap<Instance, Integer> mapa = obtenerMapa();
        for( int i=0; i<instances.numInstances(); i++){
            Instance instancia = instances.get(i);
            int ocurrencias = mapa.get(instancia);
            if( ocurrencias < k ){
                instances.delete(i);
            }
        }
        return instances;
    }

    private int mayorFrecuencia(Vector<Integer> atributos) {
        int mayor = 0;
        for( Integer i: atributos ){
            if( i>mayor )
                mayor = i;
        }
        return mayor;
    }

    private boolean esCompleto(int atributo) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private HashMap<Instance, Integer> obtenerMapa() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private boolean verificarK(Instances anonimized, int k) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
