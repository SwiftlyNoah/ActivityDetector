//
//  ContentView.swift
//  ActivityDetector
//
//  Created by Noah Brauner on 10/21/23.
//

import SwiftUI
import CoreMotion
import CoreML


struct ContentView: View {
    @ObservedObject var motionManager = MotionManager()

    var body: some View {
        VStack {
            Text(motionManager.activity)
                .frame(maxWidth: .infinity, alignment: .leading)
                .font(.largeTitle.bold())
            
            Spacer()
                .frame(height: 20)
            
            VStack(alignment: .leading) {
                ForEach(motionManager.activityProbabilities.sorted(by: >), id: \.key) { probability in
                    (
                        Text(probability.key)
                        +
                        Text(" - " + String(format: "%.3f%%", probability.value * 100))
                            .bold()
                    )
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color(UIColor.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 30))
            
            Spacer()
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

class MotionManager: ObservableObject {
    let motionManager = CMMotionManager()
    let coreMLModel: ActivityDetector?
    
    @Published var activity: String = "Unknown"
    @Published var activityProbabilities: [String : Double] = [:]
    
    var accelX: [Double] = []
    var accelY: [Double] = []
    var accelZ: [Double] = []
    var accelNorm: [Double] = []
    var gyroX: [Double] = []
    var gyroY: [Double] = []
    var gyroZ: [Double] = []
    
    let windowSize = 150  // rolling window size
    
    let abbreviatedNames = ["dws", "ups", "sit", "std", "wlk", "jog"]
    let fullNames = ["Downstairs", "Upstairs", "Sitting", "Standing", "Walking", "Jogging"]
    
    init() {
        coreMLModel = try? ActivityDetector(configuration: MLModelConfiguration())
        
        // 50 Hz updates
        motionManager.accelerometerUpdateInterval = 1 / 50
        motionManager.gyroUpdateInterval = 1 / 50
        startMotionUpdates()
    }
    
    func startMotionUpdates() {
        motionManager.startAccelerometerUpdates(to: OperationQueue.current!) { (data, error) in
            if let data = data {
                self.accelX.append(data.acceleration.x)
                self.accelY.append(data.acceleration.y)
                self.accelZ.append(data.acceleration.z)
                self.accelNorm.append(sqrt(pow(data.acceleration.x, 2) + pow(data.acceleration.y, 2) + pow(data.acceleration.z, 2)))
                self.extractFeaturesAndPredict()
            }
        }
        
        motionManager.startGyroUpdates(to: OperationQueue.current!) { (data, error) in
            if let data = data {
                self.gyroX.append(data.rotationRate.x)
                self.gyroY.append(data.rotationRate.y)
                self.gyroZ.append(data.rotationRate.z)
                self.extractFeaturesAndPredict()
            }
        }
    }
    
    func extractFeaturesAndPredict() {
        if accelX.count >= windowSize {
            let features = [accelX, accelY, accelZ, accelNorm, gyroX, gyroY, gyroZ]
            var inputFeatures: [Double] = []
            
            for feature in features {
                let lastValue = feature.last ?? 0.0  // The original non-rolling feature
                let sortedFeature = feature.suffix(windowSize).sorted()
                let mean = feature.suffix(windowSize).reduce(0, +) / Double(windowSize)
                let std = sqrt(feature.suffix(windowSize).map { pow($0 - mean, 2) }.reduce(0, +) / Double(windowSize))
                let median = sortedFeature[sortedFeature.count / 2]
                
                inputFeatures.append(contentsOf: [lastValue, mean, std, median])
            }
            
            // Prepare input for Core ML model
            let input = try? ActivityDetectorInput(dense_8_input: MLMultiArray(inputFeatures))
            
            if let input, let prediction = try? coreMLModel?.prediction(input: input) {
                let classProbabilities = prediction.classLabel_probs
                let maxProbClass = classProbabilities.max(by: { $0.value < $1.value })?.key ?? "Unknown"
                self.activityProbabilities = classProbabilities
                if let i = abbreviatedNames.firstIndex(of: maxProbClass) {
                    self.activity = fullNames[i]
                }
                else {
                    self.activity = "Unknown"
                }
            }
            
            // Remove oldest value to make room for new data
            accelX.removeFirst()
            accelY.removeFirst()
            accelZ.removeFirst()
            accelNorm.removeFirst()
            gyroX.removeFirst()
            gyroY.removeFirst()
            gyroZ.removeFirst()
        }
    }
}
