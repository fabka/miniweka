/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package miniweka;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author david
 */
public class Miniweka {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            ArrayList<Integer> lista = new ArrayList<>();
            String path = "VTargetMailWEKA.arff";
            ClusterGen cluster = new ClusterGen(path);
            Integer[] a = new Integer[] {0, 1};
            Vector<Integer> atributos = new Vector<Integer>(); 
            atributos.add(1);
            atributos.add(2);
            cluster.kmeans_test(new weka.core.EuclideanDistance(), 3, 10, 500, false, true, atributos);
            //cluster.kmeans(new weka.core.EuclideanDistance(), 3, 10, 500, false, false, Arrays.asList(0));
        } catch (IOException ex) {
            Logger.getLogger(Miniweka.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(Miniweka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
