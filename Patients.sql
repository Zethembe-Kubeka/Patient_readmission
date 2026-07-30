-- Databricks notebook source
SELECT *
FROM patients.readmission.dataset;

SELECT distinct `Admission Type`
FROM patients.readmission.dataset;

SELECT `Age`,
        `Gender`,
        `Readmission`,
        `Admission Type`
FROM patients.readmission.dataset;

SELECT `Patient ID`,
        `Age`,
        `Gender`,
        `Admission Type`,
        `Readmission`
FROM patients.readmission.dataset
WHERE `Readmission` IN ('No', 'Yes') AND `Age` BETWEEN 32 AND 65;