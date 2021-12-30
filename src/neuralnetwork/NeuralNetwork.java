/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;


import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.io.PrintWriter;
import java.util.Scanner; // Import the Scanner class to read text files
import java.util.ArrayList;

import java.util.Arrays;
import java.util.Collections;


/**
 *
 * @author Magooda
 */
public class NeuralNetwork {

    /**
     * @param args the command line arguments
     */
    static ArrayList<ArrayList> Xs = new ArrayList<>();
    static ArrayList<ArrayList> Ys = new ArrayList<>();
    static ArrayList<ArrayList> output_weights = new ArrayList<>();
    static ArrayList<ArrayList> hidden_weights = new ArrayList<>();

    static int numberofinputs, numberofhidden, numberofoutputs, numberofTrainingExamples, numberofweights;
    static double alpha = 0.5;

    public static void read_from_file(String filename) {
        try {
            File myObj = new File(filename);
            Scanner myReader = new Scanner(myObj);
            numberofinputs = myReader.nextInt();
            numberofhidden = myReader.nextInt();
            numberofoutputs = myReader.nextInt();
            numberofTrainingExamples = myReader.nextInt();
            for (int i = 0; i < numberofTrainingExamples; i++) {
                ArrayList<Double> X = new ArrayList<>();
                ArrayList<Double> Y = new ArrayList<>();
                for (int j = 0; j < numberofinputs; j++) {
                    X.add(myReader.nextDouble());
                }
                for (int j = 0; j < numberofoutputs; j++) {
                    Y.add(myReader.nextDouble());
                }
                Xs.add(X);
                Ys.add(Y);
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public static void Normalization() {
        ArrayList<ArrayList> newxs = new ArrayList<>();
        ArrayList<Double> means = new ArrayList<>(Arrays.asList(new Double[numberofinputs]));
        ArrayList<Double> Sd = new ArrayList<>(Arrays.asList(new Double[numberofinputs]));
        Collections.fill(means, 0.0);
        Collections.fill(Sd, 0.0);
        Double value = 0.0, sumTillNow = 0.0, sdTillNow = 0.0;
        for (int i = 0; i < numberofTrainingExamples; i++) {
            for (int j = 0; j < numberofinputs; j++) {
                value = (Double) Xs.get(i).get(j);
                sumTillNow = means.get(j);
                means.remove(j);
                if (i == numberofTrainingExamples - 1) {
                    means.add(j, ((value + sumTillNow) / numberofTrainingExamples));
                } else {
                    means.add(j, (value + sumTillNow));
                }
            }
        }
        for (int i = 0; i < numberofTrainingExamples; i++) {
            for (int j = 0; j < numberofinputs; j++) {
                Double mean = means.get(j);
                value = (Double) Xs.get(i).get(j);
                sdTillNow = Sd.get(j);
                Sd.remove(j);
                if (i == numberofTrainingExamples - 1) {
                    Double Val = (value - mean) * (value - mean) + sdTillNow;
                    Val = Val / numberofTrainingExamples;
                    Sd.add(j, Math.sqrt(Val));
                } else {
                    Sd.add(j, (Double) ((value - mean) * (value - mean)) + sdTillNow);
                }
            }
        }
        for (int i = 0; i < numberofTrainingExamples; i++) {
            ArrayList<Double> newx = new ArrayList<>();
            for (int j = 0; j < numberofinputs; j++) {
                Double new_x = (((Double) Xs.get(i).get(j) - means.get(j)) / Sd.get(j));
                newx.add(new_x);
            }
            newxs.add(newx);
        }
        Xs = newxs;
    }

    public static void weights_initialization() {
        numberofweights = numberofinputs * numberofhidden + numberofhidden * numberofoutputs;
        Double range = (double) (1.0 / numberofweights);
        for (int j = 0; j < numberofhidden; j++) {
            ArrayList<Double> neuron = new ArrayList<>();
            for (int i = 0; i < numberofinputs; i++) {
                neuron.add(Math.random() * (range + range) - range);
            }
            hidden_weights.add(neuron);
        }
        for (int j = 0; j < numberofoutputs; j++) {
            ArrayList<Double> neuron = new ArrayList<>();
            for (int i = 0; i < numberofhidden; i++) {
                neuron.add(Math.random() * (range + range) - range);
            }
            output_weights.add(neuron);
        }
    }

    public static ArrayList<ArrayList> FeedForward(int numOfCurrentTrainingExample) {
        ArrayList<Double> Outputs_Out = new ArrayList<>();
        ArrayList<Double> Hidden_Out = new ArrayList<>();
        ArrayList<ArrayList> AllOfTheAbove = new ArrayList<>();   //just for return 
        for (int i = 0; i < numberofhidden; i++) {
            Double H_i_in = 0.0;
            for (int j = 0; j < numberofinputs; j++) {
                H_i_in += (Double) (hidden_weights.get(i).get(j)) * ((Double) Xs.get(numOfCurrentTrainingExample).get(j));
            }
            Double Exp = Math.exp(-H_i_in);
            Double H_i_out = (1 / (1 + Exp));
            Hidden_Out.add(H_i_out);
        }
        for (int i = 0; i < numberofoutputs; i++) {
            Double Y_i_in = 0.0;
            for (int j = 0; j < numberofhidden; j++) {
                Y_i_in += (Double) (output_weights.get(i).get(j)) * ((Double) Hidden_Out.get(j));
            }
            Double Exp = Math.exp(-Y_i_in);
            Double Y_i_out = (1 / (1 + Exp));
            Outputs_Out.add(Y_i_out);
        }
        AllOfTheAbove.add(Hidden_Out);
        AllOfTheAbove.add(Outputs_Out);

        return AllOfTheAbove;
    }

    public static void back_propagation(int numOfCurrentTrainingExample, ArrayList<Double> Hiddens_Out, ArrayList<Double> Outputs_Out){
        ArrayList<Double> Output_Errors = new ArrayList<>();
        for(int i=0; i<numberofoutputs; i++){
            double out  = Outputs_Out.get(i);
            double target = (Double)Ys.get(numOfCurrentTrainingExample).get(i);
            double error = out * (1 - out) * (target - out);
            Output_Errors.add(error);
        }
        ArrayList<Double> Hidden_Errors = new ArrayList<>();
        for(int i=0; i<numberofhidden; i++){
            double out  = Hiddens_Out.get(i);
            double sum = 0.0;
            for(int j=0; j<numberofoutputs; j++){
                sum += (Output_Errors.get(j) * (Double)output_weights.get(j).get(i));
            }
            double error = out * (1 - out) * sum;
            Hidden_Errors.add(error);
        }
        update_weights(Hidden_Errors, Output_Errors,Hiddens_Out, Outputs_Out);
    }

    public static void update_weights(ArrayList<Double> Hidden_Errors,ArrayList<Double> Output_Errors,  ArrayList<Double> Hiddens_Out, ArrayList<Double> Outputs_Out){
        ArrayList<ArrayList> new_output_weights = new ArrayList<>();
        ArrayList<ArrayList> new_hidden_weights = new ArrayList<>();
        for(int i=0;  i<output_weights.size(); i++){
            ArrayList<Double> neuron = new ArrayList<>();
                for(int j=0; j<output_weights.get(i).size(); j++){
                    double old = (Double) output_weights.get(i).get(j);
                    double neww = old + alpha * Output_Errors.get(i) * Outputs_Out.get(i) ;
                    neuron.add(neww);
                }
                new_output_weights.add(neuron);
        }
        output_weights.clear();
        output_weights = new_output_weights;
        for(int i=0;  i<hidden_weights.size(); i++){
            ArrayList<Double> neuron = new ArrayList<>();
                for(int j=0; j<hidden_weights.get(i).size(); j++){
                    double old = (Double) hidden_weights.get(i).get(j);
                    double neww = old + alpha * Hidden_Errors.get(i) * Hiddens_Out.get(i) ;
                    neuron.add(neww);
                }
                new_hidden_weights.add(neuron);
        }
        hidden_weights.clear();
        hidden_weights = new_hidden_weights;
    }
   
    public static double calc_MSE(ArrayList<ArrayList> Outputs_AllEx){
        double MSE = 0.0;
        ArrayList<Double> sumError = new ArrayList<>();
        for(int i=0; i<Ys.size(); i++){
                double sum = 0.0;
                for(int j=0; j<Ys.get(i).size(); j++){
                        double diff = (Double) Ys.get(i).get(j) - (Double) Outputs_AllEx.get(i).get(j);
                        double sq = Math.pow(diff, 2) ;
                        sum += sq;
                }
                sumError.add(0.5*sum);
        }
        double sum = 0.0;
        for(int i=0; i<sumError.size(); i++){
                sum += sumError.get(i);
        }
        MSE =  sum/numberofTrainingExamples;

        return MSE;
    }
  
    public static void first_program(){
        read_from_file("train.txt");
        if(numberofTrainingExamples > 1){
            Normalization();
        }
        weights_initialization();
        double MSE = 0.0;
        for (int iter=0; iter < 500; iter++){
            ArrayList<ArrayList> Outputs_AllEx = new ArrayList<>();
            for (int j = 0; j < numberofTrainingExamples; j++) {
                ArrayList<Double> Outputs_OneEx = new ArrayList<>();
                ArrayList<Double> Hiddens_OneEx = new ArrayList<>();
                ArrayList<ArrayList> Both_Hidden_Out = new ArrayList<>();
                Both_Hidden_Out = FeedForward(j);
                Outputs_OneEx = Both_Hidden_Out.get(1);
                Outputs_AllEx.add(Both_Hidden_Out.get(1));
                Hiddens_OneEx = Both_Hidden_Out.get(0);
                back_propagation(j, Hiddens_OneEx,Outputs_OneEx);
            }
            MSE = calc_MSE(Outputs_AllEx);
        }
        System.out.println("Training MSE = " + MSE);
   }

    public static void write_weights_to_file(){
        File f = new File("weights.txt");
        try (PrintWriter pw = new PrintWriter(f)) {
                pw.println("Output Weights : ");
                for(int i=0; i<output_weights.size(); i++){
                    pw.println(output_weights.get(i));
                }
                pw.println();
                pw.println("Hidden Weights : ");
                for(int i=0; i<hidden_weights.size(); i++){
                    pw.println(hidden_weights.get(i));
                }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void second_program(){
        read_from_file("input.txt");
        if(numberofTrainingExamples > 1){
            Normalization();
        }
        ArrayList<ArrayList> Outputs_AllEx = new ArrayList<>();
        for (int j = 0; j < numberofTrainingExamples; j++) {
                ArrayList<ArrayList> Both_Hidden_Out = new ArrayList<>();
                Both_Hidden_Out = FeedForward(j);
                Outputs_AllEx.add(Both_Hidden_Out.get(1));
        }
        double MSE = calc_MSE(Outputs_AllEx);
        System.out.println("Input MSE = " + MSE);
   }
    public static void main(String[] args) {
        System.out.println("First Program : ");
        first_program();
        write_weights_to_file();
        Xs.clear();
        Ys.clear();
        System.out.println("Second Program : ");
        second_program();
    }
}
