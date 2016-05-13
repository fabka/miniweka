/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package miniweka;

import weka.core.*;
import weka.filters.unsupervised.instance.RemoveFrequentValues;

/**
 *
 * @author Monica
 */
public class DataFly {

    Instances instances;
    
    public DataFly( Instances instances) {
        this.instances = instances;
    }
    
    public void test(){
        RemoveFrequentValues rfv = new RemoveFrequentValues();
        rfv.
    }
    
        
}
